import asyncio
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

from telegram import Update, Message, File
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

# Папки
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Память: "ждём описание" для последнего файла от пользователя
# user_id -> {"expires": datetime, "record": {метаданные без description}}
PENDING_DESC = {}

DESC_TIMEOUT = timedelta(minutes=5)  # сколько ждём текст-описание после файла

def write_jsonl(record: dict, path: Path = DATA_DIR / "uploads.jsonl"):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Присылай файл. Описание можно указать в подписи к файлу "
        "или отправить текстом сразу после файла (в течение 5 минут)."
    )

def ensure_user_dir(user_id: int) -> Path:
    p = UPLOAD_DIR / str(user_id)
    p.mkdir(parents=True, exist_ok=True)
    return p

def sniff_file_kind(msg: Message) -> str:
    if msg.document: return "document"
    if msg.photo:    return "photo"
    if msg.video:    return "video"
    if msg.audio:    return "audio"
    if msg.voice:    return "voice"
    return "unknown"

async def download_any_file(msg: Message) -> tuple[Path, dict]:
    """
    Скачивает файл из Message и возвращает (путь_к_файлу, метаданные_без_description).
    """
    user = msg.from_user
    user_dir = ensure_user_dir(user.id)

    file_obj: File | None = None
    filename = None
    kind = sniff_file_kind(msg)

    if msg.document:
        file_obj = await msg.document.get_file()
        filename = msg.document.file_name or f"document_{msg.document.file_unique_id}"
    elif msg.photo:
        # Берём последнюю (самую большую) фотографию
        file_obj = await msg.photo[-1].get_file()
        filename = f"photo_{msg.photo[-1].file_unique_id}.jpg"
    elif msg.video:
        file_obj = await msg.video.get_file()
        filename = msg.video.file_name or f"video_{msg.video.file_unique_id}.mp4"
    elif msg.audio:
        file_obj = await msg.audio.get_file()
        filename = msg.audio.file_name or f"audio_{msg.audio.file_unique_id}.mp3"
    elif msg.voice:
        file_obj = await msg.voice.get_file()
        filename = f"voice_{msg.voice.file_unique_id}.ogg"
    else:
        raise RuntimeError("Unsupported message type")

    # Уникализируем имя
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{stamp}_{filename}"
    dst = user_dir / safe_name
    await file_obj.download_to_drive(dst)

    # Базовые метаданные
    record = {
        "user_id": user.id,
        "username": user.username,
        "first_name": user.first_name,
        "kind": kind,
        "file_id": file_obj.file_id,
        "file_unique_id": getattr(getattr(msg, kind, None), "file_unique_id", None) if kind != "photo" else msg.photo[-1].file_unique_id,
        "original_filename": filename,
        "saved_path": str(dst.relative_to(BASE_DIR)),
        "message_id": msg.message_id,
        "chat_id": msg.chat_id,
        "media_group_id": msg.media_group_id,  # пригодится, если пришёл альбом
        "caption": msg.caption,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "description": None,  # заполним позже, если придёт текст
    }
    return dst, record

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    try:
        _, record = await download_any_file(msg)

        # Если уже есть caption — сразу сохраняем полноценную запись
        if record["caption"]:
            record["description"] = record["caption"]
            write_jsonl(record)
            await msg.reply_text("Файл сохранён ✅ Описание взято из подписи.")
            return

        # Иначе ждём текст в следующем сообщении
        PENDING_DESC[msg.from_user.id] = {
            "expires": datetime.utcnow() + DESC_TIMEOUT,
            "record": record,
        }
        await msg.reply_text(
            "Файл сохранён ✅ Пришли текст-описание одним сообщением (до 5 минут)."
        )
    except Exception as e:
        await msg.reply_text(f"Не получилось обработать файл: {e}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    uid = msg.from_user.id
    pend = PENDING_DESC.get(uid)

    # Чистим протухшие ожидания
    if pend and pend["expires"] < datetime.utcnow():
        PENDING_DESC.pop(uid, None)
        pend = None

    if pend:
        record = pend["record"]
        record["description"] = msg.text
        write_jsonl(record)
        PENDING_DESC.pop(uid, None)
        await msg.reply_text("Описание получено 📝 Запись сохранена.")
    else:
        await msg.reply_text(
            "Это текст. Если хочешь привязать его как описание, сначала пришли файл."
        )

async def main():
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Установи переменную окружения BOT_TOKEN")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start_cmd))

    file_filters = (
        filters.Document.ALL
        | filters.PHOTO
        | filters.VIDEO
        | filters.AUDIO
        | filters.VOICE
    )
    app.add_handler(MessageHandler(file_filters, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Запуск
    print("Bot is running...")
    await app.run_polling(close_loop=False)

if __name__ == "__main__":
    asyncio.run(main())