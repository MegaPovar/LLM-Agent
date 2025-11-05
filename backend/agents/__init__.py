from pathlib import Path
import json

class AgentBase:
    name = "base"

    def run(self, ctx: dict) -> dict:
        """Каждый агент переопределяет этот метод"""
        raise NotImplementedError

    def save_context(self, ctx: dict):
        """Сохраняет контекст после обработки"""
        out_dir = Path(ctx["files"]["out_dir"])
        with open(out_dir / "context.json", "w", encoding="utf-8") as f:
            json.dump(ctx, f, ensure_ascii=False, indent=2)