import pandas as pd
from . import AgentBase

class QualityAgent(AgentBase):
    name = "quality"

    def run(self, ctx: dict) -> dict:
        df = pd.read_csv(ctx["files"]["dataset"]) if ctx["files"]["dataset"].endswith(".csv") \
             else pd.read_excel(ctx["files"]["dataset"])

        # Базовая информация
        ctx["meta"]["shape"] = df.shape
        ctx["metrics"]["null_counts"] = df.isna().sum().to_dict()
        ctx["metrics"]["duplicates"] = int(df.duplicated().sum())

        # Простая оценка качества
        bad_cols = [c for c, v in ctx["metrics"]["null_counts"].items() if v / len(df) > 0.3]
        ctx["findings"]["quality"] = {
            "high_nul   l_columns": bad_cols,
            "duplicate_rows": ctx["metrics"]["duplicates"]
        }

        ctx["brief"] = f"Quality check: {len(bad_cols)} columns have >30% missing values."
        self.save_context(ctx)
        return ctx