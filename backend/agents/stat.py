import pandas as pd
import numpy as np
from scipy import stats
from . import AgentBase

class StatAgent(AgentBase):
    name = "stat"

    def run(self, ctx: dict) -> dict:
        dataset_path = ctx["files"]["dataset"]

        # безопасное чтение
        if dataset_path.endswith(".csv"):
            df = pd.read_csv(dataset_path)
        else:
            df = pd.read_excel(dataset_path)

        # числовые/категориальные колонки
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        results = {}

        # --- 1) Корреляции Пирсона по числовым ---
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().round(3)

            # сохраняем полную матрицу в виде обычного dict (OK для JSON)
            results["pearson_correlation_matrix"] = corr.to_dict()

            # берём только верхний треугольник без диагонали
            mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
            corr_upper = corr.where(mask)

            # плоский список пар без NaN
            flat = (
                corr_upper.stack()                      # MultiIndex (col1, col2) -> value
                .abs()
                .sort_values(ascending=False)
            )

            # top-N как список записей (JSON-friendly)
            top_n = []
            for (c1, c2), val in flat.head(5).items():
                top_n.append({
                    "col1": c1,
                    "col2": c2,
                    "corr_abs": float(val),
                    "corr": float(corr.loc[c1, c2]),
                })
            results["top_correlations"] = top_n
        else:
            results["pearson_correlation_matrix"] = {}
            results["top_correlations"] = []

        # --- 2) Пробный t-test: первая пара (num, cat) с 2 группами ---
        if len(numeric_cols) >= 1 and len(cat_cols) >= 1:
            num = numeric_cols[0]
            cat = cat_cols[0]
            try:
                groups = df[cat].dropna().unique()
                if len(groups) >= 2:
                    g1, g2 = groups[:2]
                    x = df.loc[df[cat] == g1, num].dropna()
                    y = df.loc[df[cat] == g2, num].dropna()
                    if len(x) > 2 and len(y) > 2:
                        t_stat, p_val = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
                        results["t_test"] = {
                            "numeric_column": num,
                            "group_column": cat,
                            "groups": [str(g1), str(g2)],
                            "n1": int(len(x)),
                            "n2": int(len(y)),
                            "t_statistic": round(float(t_stat), 4),
                            "p_value": round(float(p_val), 6),
                        }
            except Exception as e:
                results["t_test_error"] = str(e)

        # --- 3) Итог в контекст ---
        ctx.setdefault("metrics", {})
        ctx["metrics"]["statistical_tests"] = results
        ctx["brief"] = "Stat: correlations computed, top pairs listed, t-test attempted."
        self.save_context(ctx)
        return ctx