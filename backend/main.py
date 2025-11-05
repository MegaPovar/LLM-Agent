from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
import uuid, json, os

# --- env ---
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions")

# --- FastAPI app ---
app = FastAPI(title="Multi-Agent Data Analysis API")

# --- FS paths ---
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

# --- Agents ---
from agents.quality import QualityAgent
from agents.describe import DescribeAgent
from agents.stat import StatAgent
# подключи остальные при появлении:
# from agents.research import ResearchAgent
# from agents.eval import EvalAgent
# from agents.viz import VizAgent
# from agents.triviality import TrivialityAgent
# from agents.biz import BizAgent

AGENTS = {
    "quality": QualityAgent(),
    "describe": DescribeAgent(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL),
    "stat":     StatAgent(),
    # "research": ResearchAgent(),
    # "eval":     EvalAgent(),
    # "viz":      VizAgent(),
    # "triviality": TrivialityAgent(),
    # "biz":        BizAgent(),
}
PIPELINE_ORDER = [  # порядок запуска для /run-all
    "quality", "describe", "stat",
    # "research", "eval", "viz", "triviality", "biz"
]

# --- helpers ---
def ctx_path_for(task_id: str) -> Path:
    return ARTIFACTS_DIR / task_id / "context.json"

def load_ctx(task_id: str) -> dict:
    p = ctx_path_for(task_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Context not found")
    return json.loads(p.read_text(encoding="utf-8"))

def save_ctx(task_id: str, ctx: dict) -> None:
    p = ctx_path_for(task_id)
    p.write_text(json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8")

# --- models ---
class RunRequest(BaseModel):
    task_id: str

# --- endpoints ---
@app.get("/ping")
def ping():
    return {"ok": True, "agents": list(AGENTS.keys())}

@app.post("/start")
async def start_analysis(file: UploadFile, prompt: str = Form("")):
    """
    Инициализирует задачу: сохраняет датасет и контекст. Возвращает task_id.
    """
    task_id = str(uuid.uuid4())
    out_dir = ARTIFACTS_DIR / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dir / file.filename
    with open(dataset_path, "wb") as f:
        f.write(await file.read())

    ctx = {
        "task_id": task_id,
        "prompt": prompt,
        "files": {
            "dataset": str(dataset_path),
            "out_dir": str(out_dir),
            "artifacts": []
        },
        "meta": {},
        "metrics": {},
        "findings": {},
        "insights": [],
        "progress": []
    }
    save_ctx(task_id, ctx)
    return {"task_id": task_id, "message": "Dataset uploaded, ready to process."}

@app.post("/run/{agent_name}")
def run_agent(agent_name: str, body: RunRequest):
    """
    Запускает конкретного агента и сохраняет обновлённый контекст.
    """
    if agent_name not in AGENTS:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    ctx = load_ctx(body.task_id)

    agent = AGENTS[agent_name]
    updated_ctx = agent.run(ctx)  # агент сам добавляет brief/artifacts в ctx
    updated_ctx.setdefault("progress", []).append(agent_name)

    save_ctx(body.task_id, updated_ctx)
    return {
        "task_id": body.task_id,
        "agent": agent_name,
        "brief": updated_ctx.get("brief", ""),
        "artifacts": updated_ctx["files"].get("artifacts", []),
        "progress": updated_ctx.get("progress", [])
    }

@app.post("/run-all")
def run_all(body: RunRequest):
    """
    Прогоняет пайплайн целиком в порядке PIPELINE_ORDER.
    """
    ctx = load_ctx(body.task_id)
    for name in PIPELINE_ORDER:
        agent = AGENTS.get(name)
        if not agent:
            continue
        ctx = agent.run(ctx)
        ctx.setdefault("progress", []).append(name)
        save_ctx(body.task_id, ctx)
    return {"task_id": body.task_id, "message": "Pipeline completed", "progress": ctx.get("progress", [])}

@app.get("/status/{task_id}")
def status(task_id: str):
    """
    Возвращает текущий контекст (метаданные, метрики, инсайты, список артефактов).
    """
    return load_ctx(task_id)

@app.get("/file/{task_id}/{filename}")
def get_file(task_id: str, filename: str):
    """
    Отдаёт файл-артефакт по имени.
    """
    file_path = ARTIFACTS_DIR / task_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.get("/agents")
def list_agents():
    return {"available": list(AGENTS.keys()), "order": PIPELINE_ORDER}