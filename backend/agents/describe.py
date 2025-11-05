import pandas as pd
import json
import requests
from . import AgentBase

class DescribeAgent(AgentBase):
    name = "describe"

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/chat/completions"):
        self.api_key = api_key
        self.base_url = base_url

    def get_dataframe_info(self, df: pd.DataFrame) -> dict:
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        categorical_columns = df.select_dtypes(exclude="number").columns.tolist()
        return {
            "shape": tuple(map(int, df.shape)),
            "columns": list(df.columns),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().astype(int).to_dict(),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "sample_data": df.head(5).to_dict("records"),
        }

    def analyze_dataframe(self, dataframe_info: dict, stat_results: dict | None = None) -> str:
        # Жёсткий промпт + формат под Telegram-HTML
        system_prompt = (
            "Ты — аналитик данных. Пиши строго по фактам из переданного JSON. "
            "Не придумывай значения и поля, которых нет. Кратко и структурировано."
        )

        # Вшиваем факты и (если есть) результаты stat-агента
        user_prompt = f"""
ФАКТЫ О ДАТАСЕТЕ (JSON):
{json.dumps(dataframe_info, ensure_ascii=False)}

РЕЗУЛЬТАТЫ СТАТИСТИКИ (если есть):
{json.dumps(stat_results or {}, ensure_ascii=False)}

СФОРМИРУЙ ОТВЕТ НА РУССКОМ в формате Telegram HTML:

<b>О чём датасет</b>
• 1–2 чётких предложения, только по колонкам/статистике.

<b>Ключевые сигналы из данных</b>
• 4–7 пунктов с конкретными числами/фактами (кардинальности, диапазоны, пропуски, дубликаты, корреляции).

<b>Подозрительные моменты / риски качества</b>
• Пропуски, выбросы, дубликаты, странные кардинальности, проблемы с датами (если есть).

<b>Что считать дальше</b>
• 3–6 следующих шагов анализа (метрики/тесты/графики) с упоминанием колонок.

ТРЕБОВАНИЯ К ФОРМАТУ:
— Используй только Telegram HTML: <b>, <i>, <u>, <code>, <s>. Без Markdown, без #, без **, без ```.
— Списки оформляй маркерами "• " в начале строки.
— Без таблиц и ссылок. До ~1500 символов.
""".strip()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 900,
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            # Фоллбек без LLM, чтобы пайплайн не падал
            cols = ", ".join(dataframe_info.get("columns", [])[:10])
            return (
                f"<b>О чём датасет</b>\n"
                f"• {dataframe_info.get('shape', ['?','?'])[0]} строк(и), "
                f"{dataframe_info.get('shape', ['?','?'])[1]} колонок ({cols}{'…' if len(dataframe_info.get('columns', []))>10 else ''}).\n\n"
                f"<b>Ключевые сигналы из данных</b>\n"
                f"• Числовые: {dataframe_info.get('numeric_columns', [])}\n"
                f"• Категориальные: {dataframe_info.get('categorical_columns', [])}\n"
                f"• Пропуски по колонкам: часть полей может содержать пропуски\n\n"
                f"<b>Что считать дальше</b>\n"
                f"• Корреляции по числовым, частоты по категориям, проверка выбросов, графики распределений."
            )

    def run(self, ctx: dict) -> dict:
        dataset_path = ctx["files"]["dataset"]
        df = pd.read_csv(dataset_path) if dataset_path.endswith(".csv") else pd.read_excel(dataset_path)

        dataframe_info = self.get_dataframe_info(df)
        # Берём результаты статистики, если их уже насчитал StatAgent
        stat_results = ctx.get("metrics", {}).get("statistical_tests", {})

        text_summary = self.analyze_dataframe(dataframe_info, stat_results)

        # Безопасные записи в контекст
        ctx.setdefault("findings", {})
        ctx.setdefault("insights", [])
        ctx.setdefault("metrics", {})

        ctx["findings"]["describe"] = dataframe_info
        ctx["insights"].append(text_summary)
        ctx["brief"] = "Dataset analyzed and described (Telegram-HTML)."
        self.save_context(ctx)
        return ctx