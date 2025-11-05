import pandas as pd
import numpy as np
from scipy import stats
from . import AgentBase

class StatAgent(AgentBase):
    name = "stat"

    def run(self, ctx: dict) -> dict:
        dataset_path = ctx["files"]["dataset"]
        df = pd.read_csv(dataset_path) if dataset_path.endswith(".csv") else pd.read_excel(dataset_path)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        results = {}

        # --- 1. Корреляции Пирсона ---
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().round(3)
            results["pearson_correlation"] = corr.to_dict()
            top_corr = corr.abs().unstack().sort_values(ascending=False)
            top_corr = top_corr[top_corr < 1].head(5)
            results["top_correlations"] = top_corr.to_dict()
        else:
            results["pearson_correlation"] = {}
            results["top_correlations"] = {}

        # --- 2. Пример: t-test между двумя группами (если найдётся подходящее поле) ---
        if len(numeric_cols) >= 1 and len(cat_cols) >= 1:
            num = numeric_cols[0]
            cat = cat_cols[0]
            try:
                groups = df[cat].dropna().unique()[:2]
                if len(groups) == 2:
                    g1 = df[df[cat] == groups[0]][num].dropna()
                    g2 = df[df[cat] == groups[1]][num].dropna()
                    if len(g1) > 2 and len(g2) > 2:
                        t_stat, p_val = stats.ttest_ind(g1, g2)
                        results["t_test"] = {
                            "numeric_column": num,
                            "group_column": cat,
                            "groups": list(groups),
                            "t_statistic": round(float(t_stat), 4),
                            "p_value": round(float(p_val), 4)
                        }
            except Exception as e:
                results["t_test"] = {"error": str(e)}

        # --- 3. Итог ---
        ctx["metrics"]["statistical_tests"] = results
        ctx["brief"] = "Statistical analysis complete: correlations and sample tests calculated."
        self.save_context(ctx)
        return ctx