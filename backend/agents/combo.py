import pandas as pd
import numpy as np
import requests
from pathlib import Path
from . import AgentBase


class ComboAgent(AgentBase):
    """
    Мега-агент: Оценка качества + Тривиальность гипотез + Бизнес-советы.
    """
    name = "combo"

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    # --- блок оценки качества ---
    def get_dataset_assessment(self, df: pd.DataFrame) -> dict:
        null_percent = df.isna().mean().to_dict()
        duplicate_rows = int(df.duplicated().sum())

        completeness = round(1 - (sum(df.isna().sum()) / (df.shape[0] * df.shape[1])), 3)

        quality_score = max(1, round(10 - (duplicate_rows / 10) - (sum(null_percent.values()) * 2), 1))
        quality_score = min(10, quality_score)

        characteristics = {
            "num_rows": df.shape[0],
            "num_columns": df.shape[1],
            "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),
            "categorical_columns": df.select_dtypes(exclude=np.number).columns.tolist(),
        }

        potential_issues = [col for col, p in null_percent.items() if p > 0.3]

        return {
            "completeness": completeness,
            "quality_score": quality_score,
            "potential_issues": potential_issues,
            "duplicate_rows": duplicate_rows,
            "data_characteristics": characteristics,
        }

    # --- DeepSeek LLM ---
    def analyze_business_potential(self, dataset_info: dict, user_prompt: str) -> str:
        sys_prompt = f"""
Ты — эксперт по данным, бизнес-аналитике и формулированию гипотез.
Проанализируй датасет по трём направлениям:

1) Оценка качества данных (строго и по делу)
2) Тривиальность гипотез (что очевидно, что нет)
3) Бизнес-советы (какие метрики/направления имеет смысл исследовать)

Не используй markdown.
Не используй теги HTML.
Пиши простым текстом.
Длина до 1500 символов.
        
ДАННЫЕ:
{dataset_info}
"""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt or "Проанализируй датасет."}
            ],
            "temperature": 0.1,
            "max_tokens": 1500,
        }

        try:
            r = requests.post(self.base_url, json=payload, headers=headers, timeout=40)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

        except Exception:
            return self.fallback_analysis(dataset_info)

    # --- Фоллбек ---
    def fallback_analysis(self, dataset_info: dict) -> str:
        q = dataset_info.get("quality_assessment", {})
        return f"""
ОЦЕНКА КАЧЕСТВА: score {q.get('quality_score')} из 10.
Проблемные места: {', '.join(q.get('potential_issues', []))}

ТРИВИАЛЬНОСТЬ: Требуется больше данных для уверенного анализа.

БИЗНЕС-СОВЕТЫ:
— исследовать зависимости числовых и категориальных признаков
— проверить распределения по группам
— построить сегменты
"""

    # --- главный метод ---
    def run(self, ctx: dict) -> dict:
        dataset_path = ctx["files"]["dataset"]

        df = pd.read_csv(dataset_path) if dataset_path.endswith(".csv") else pd.read_excel(dataset_path)

        assessment = self.get_dataset_assessment(df)

        dataset_info = {
            "basic_info": {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "types": df.dtypes.astype(str).to_dict()
            },
            "quality_assessment": assessment,
            "sample_statistics": df.describe(include="all").to_dict()
        }

        analysis = self.analyze_business_potential(dataset_info, ctx.get("prompt", ""))

        ctx.setdefault("findings", {})
        ctx.setdefault("metrics", {})
        ctx.setdefault("insights", [])

        ctx["findings"]["combo"] = assessment
        ctx["metrics"]["quality_score"] = assessment["quality_score"]
        ctx["insights"].append(analysis)
        ctx["brief"] = "Комбинированный анализ: качество + гипотезы + бизнес."

        # сохраняем txt-отчет
        out = Path(ctx["files"]["out_dir"]) / "combo_report.txt"
        out.write_text(analysis, encoding="utf-8")
        ctx["files"]["artifacts"].append(str(out))

        self.save_context(ctx)
        return ctx