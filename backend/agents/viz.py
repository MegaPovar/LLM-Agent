import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")  # без GUI
import matplotlib.pyplot as plt

from . import AgentBase


class VizAgent(AgentBase):
    """
    Агент визуализации.

    Логика:
    - даём LLM описание датасета + список доступных функций-рисовалок;
    - LLM через function calling выбирает, какие графики построить и с какими аргументами;
    - Python реально строит эти графики и сохраняет в artifacts;
    - в ctx добавляем пути к файлам и краткий текст про то, что построили.
    """
    name = "viz"

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/chat/completions"):
        self.api_key = api_key
        self.base_url = base_url

    # ---------- служебное описание датасета ----------

    def _describe_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric = df.select_dtypes(include="number").columns.tolist()
        categorical = df.select_dtypes(exclude="number").columns.tolist()
        return {
            "shape": list(map(int, df.shape)),
            "columns": list(df.columns),
            "numeric_columns": numeric,
            "categorical_columns": categorical,
            "head": df.head(5).to_dict("records"),
        }

    # ---------- функции-рисовалки (будут дергаться через tool calls) ----------

    def _plot_histogram(self, df: pd.DataFrame, column: str, out_dir: Path,
                        bins: int | None = None) -> Tuple[str, str]:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not in dataframe")

        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            raise ValueError(f"Column '{column}' has no numeric data")

        bins = bins or 20
        fig, ax = plt.subplots()
        ax.hist(series, bins=bins)
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        fig.tight_layout()

        filename = f"hist_{column}.png"
        path = out_dir / filename
        fig.savefig(path)
        plt.close(fig)

        desc = f"Гистограмма числового признака '{column}' (bins={bins})."
        return str(path), desc

    def _plot_scatter(self, df: pd.DataFrame, x: str, y: str, out_dir: Path) -> Tuple[str, str]:
        if x not in df.columns or y not in df.columns:
            raise ValueError(f"Columns '{x}' or '{y}' not in dataframe")

        x_ser = pd.to_numeric(df[x], errors="coerce").dropna()
        y_ser = pd.to_numeric(df[y], errors="coerce").dropna()

        # Выравниваем по общим индексам
        common_idx = x_ser.index.intersection(y_ser.index)
        x_ser = x_ser.loc[common_idx]
        y_ser = y_ser.loc[common_idx]

        if len(x_ser) < 3:
            raise ValueError("Not enough numeric data for scatter plot")

        fig, ax = plt.subplots()
        ax.scatter(x_ser, y_ser, s=10)
        ax.set_title(f"Scatter: {x} vs {y}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        fig.tight_layout()

        filename = f"scatter_{x}_vs_{y}.png"
        path = out_dir / filename
        fig.savefig(path)
        plt.close(fig)

        desc = f"Точечная диаграмма '{x}' vs '{y}'."
        return str(path), desc

    def _plot_boxplot(self, df: pd.DataFrame, column: str, by: str | None, out_dir: Path) -> Tuple[str, str]:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not in dataframe")

        series = pd.to_numeric(df[column], errors="coerce")
        if by is not None:
            if by not in df.columns:
                raise ValueError(f"Group column '{by}' not in dataframe")
            groups = df[by].astype(str)
            fig, ax = plt.subplots()
            data = [series[groups == g].dropna() for g in groups.unique()]
            ax.boxplot(data, labels=list(groups.unique()))
            ax.set_title(f"Boxplot of {column} by {by}")
            ax.set_xlabel(by)
            ax.set_ylabel(column)
            filename = f"box_{column}_by_{by}.png"
            desc = f"Boxplot '{column}' по группам '{by}'."
        else:
            ser_clean = series.dropna()
            if ser_clean.empty:
                raise ValueError(f"Column '{column}' has no numeric data")
            fig, ax = plt.subplots()
            ax.boxplot(ser_clean)
            ax.set_title(f"Boxplot of {column}")
            ax.set_ylabel(column)
            filename = f"box_{column}.png"
            desc = f"Boxplot числового признака '{column}'."

        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path)
        plt.close(fig)
        return str(path), desc

    # ---------- описание tools для function calling ----------

    def _tools_schema(self) -> List[Dict[str, Any]]:
        """Схема функций для DeepSeek tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "plot_histogram",
                    "description": "Построить гистограмму числовой колонки.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "column": {
                                "type": "string",
                                "description": "Название числовой колонки для гистограммы."
                            },
                            "bins": {
                                "type": "integer",
                                "description": "Количество корзин (bins), по умолчанию 20.",
                                "minimum": 5,
                                "maximum": 100
                            }
                        },
                        "required": ["column"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "plot_scatter",
                    "description": "Построить scatter plot между двумя числовыми колонками.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "string",
                                "description": "Название числовой колонки по оси X."
                            },
                            "y": {
                                "type": "string",
                                "description": "Название числовой колонки по оси Y."
                            }
                        },
                        "required": ["x", "y"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "plot_boxplot",
                    "description": "Построить boxplot для числовой колонки, опционально по группам.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "column": {
                                "type": "string",
                                "description": "Название числовой колонки."
                            },
                            "by": {
                                "type": ["string", "null"],
                                "description": "Категориальная колонка для группировки (или null)."
                            }
                        },
                        "required": ["column"]
                    }
                }
            },
        ]

    # ---------- вызов DeepSeek с tools и исполнение tool_calls ----------

    def _ask_llm_for_plots(
        self,
        df_info: Dict[str, Any],
        numeric_cols: List[str],
        cat_cols: List[str],
    ) -> Dict[str, Any]:
        """
        Делает запрос к DeepSeek с tools, чтобы получить tool_calls.
        """
        system_prompt = (
            "Ты — помощник по визуализации данных. "
            "Тебе дано описание датасета и список доступных функций визуализации. "
            "Твоя задача — выбрать до 3 осмысленных графиков и вызвать соответствующие функции. "
            "Не пиши обычный текст-ответ, используй только tool calls."
        )

        user_prompt = f"""
ДАННЫЕ О ДАТАСЕТЕ:
{json.dumps(df_info, ensure_ascii=False, indent=2)}

ДОСТУПНЫЕ ЧИСЛОВЫЕ КОЛОНКИ:
{numeric_cols}

ДОСТУПНЫЕ КАТЕГОРИАЛЬНЫЕ КОЛОНКИ:
{cat_cols}

СДЕЛАЙ:
- Выбери не более 3 визуализаций.
- Используй функции plot_histogram / plot_scatter / plot_boxplot.
- Для plot_scatter выбирай только числовые колонки.
- Для plot_boxplot можешь использовать группировку по одной категориальной колонке (или без grouping).
- Не добавляй обычный текст — только tool calls.
""".strip()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: Dict[str, Any] = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "tools": self._tools_schema(),
            "tool_choice": "auto",
            "temperature": 0.1,
            "max_tokens": 512,
        }

        resp = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()

    # ---------- основной метод агента ----------

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        dataset_path = ctx["files"]["dataset"]
        df = pd.read_csv(dataset_path) if dataset_path.endswith(".csv") else pd.read_excel(dataset_path)

        out_dir = Path(ctx["files"]["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        df_info = self._describe_df(df)
        numeric_cols = df_info["numeric_columns"]
        cat_cols = df_info["categorical_columns"]

        # гарантируем ключи
        ctx.setdefault("files", {})
        ctx["files"].setdefault("artifacts", [])
        ctx.setdefault("insights", [])
        ctx.setdefault("metrics", {})
        ctx.setdefault("findings", {})

        generated: List[str] = []
        descriptions: List[str] = []

        try:
            llm_response = self._ask_llm_for_plots(df_info, numeric_cols, cat_cols)
            message = llm_response["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []

            for call in tool_calls:
                fn_name = call["function"]["name"]
                raw_args = call["function"].get("arguments") or "{}"
                try:
                    args = json.loads(raw_args)
                except Exception:
                    args = {}

                try:
                    if fn_name == "plot_histogram":
                        path, desc = self._plot_histogram(
                            df,
                            column=args.get("column"),
                            out_dir=out_dir,
                            bins=args.get("bins"),
                        )
                    elif fn_name == "plot_scatter":
                        path, desc = self._plot_scatter(
                            df,
                            x=args.get("x"),
                            y=args.get("y"),
                            out_dir=out_dir,
                        )
                    elif fn_name == "plot_boxplot":
                        path, desc = self._plot_boxplot(
                            df,
                            column=args.get("column"),
                            by=args.get("by"),
                            out_dir=out_dir,
                        )
                    else:
                        continue

                    if path not in ctx["files"]["artifacts"]:
                        ctx["files"]["artifacts"].append(path)
                    generated.append(path)
                    descriptions.append(desc)
                except Exception as e:
                    # не валим пайплайн из-за одной неудачной картинки
                    descriptions.append(f"Ошибка при построении {fn_name}: {e}")

        except Exception as e:
            # fallback: если что-то с LLM/tool calling, делаем простые гистограммы
            if numeric_cols:
                col = numeric_cols[0]
                try:
                    path, desc = self._plot_histogram(df, column=col, out_dir=out_dir)
                    if path not in ctx["files"]["artifacts"]:
                        ctx["files"]["artifacts"].append(path)
                    generated.append(path)
                    descriptions.append(
                        f"LLM не сработал, но построена базовая гистограмма по '{col}'. Ошибка: {e}"
                    )
                except Exception as ee:
                    descriptions.append(f"Не удалось построить даже базовую гистограмму: {ee}")
            else:
                descriptions.append(f"LLM/tool calling дал ошибку, и нет числовых колонок для fallback: {e}")

        if generated:
            ctx["insights"].append(
                "Сгенерированы визуализации:\n" + "\n".join(f"- {d}" for d in descriptions)
            )
            ctx["brief"] = "Визуализация датасета выполнена."
        else:
            ctx["insights"].append(
                "Агент визуализации не смог построить ни одного графика (нет подходящих данных или все вызовы упали)."
            )
            ctx["brief"] = "Визуализация не выполнена."

        self.save_context(ctx)
        return ctx