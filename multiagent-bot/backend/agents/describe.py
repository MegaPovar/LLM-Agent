import pandas as pd
import json
import requests
from io import StringIO
from . import AgentBase

class DescribeAgent(AgentBase):
    name = "describe"

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/chat/completions"):
        self.api_key = api_key
        self.base_url = base_url

    def get_dataframe_info(self, df):
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        categorical_columns = df.select_dtypes(exclude="number").columns.tolist()
        description = {
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "sample_data": df.head(5).to_dict("records"),
        }
        return description

    def analyze_dataframe(self, dataframe_info):
        system_prompt = f"""
        Ты опытный дата-аналитик. Тебе нужно сделать краткое описание датасета
        для начинающего аналитика: что, вероятно, он содержит, какие особенности видны,
        какие возможные направления анализа.

        ИНФОРМАЦИЯ О ДАТАСЕТЕ:
        - Размер: {dataframe_info['shape']}
        - Колонки: {dataframe_info['columns']}
        - Типы данных: {json.dumps(dataframe_info['data_types'], ensure_ascii=False, indent=2)}
        - Пропущенные значения: {json.dumps(dataframe_info['missing_values'], ensure_ascii=False, indent=2)}
        - Числовые колонки: {dataframe_info['numeric_columns']}
        - Категориальные колонки: {dataframe_info['categorical_columns']}

        Первые 5 строк данных:
        {json.dumps(dataframe_info['sample_data'], ensure_ascii=False, indent=2)}
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Дай краткое описание и предположение о назначении датасета."}
            ],
            "temperature": 0.2,
            "max_tokens": 800
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            return f"Ошибка при обращении к DeepSeek API: {e}"

    def run(self, ctx: dict) -> dict:
        dataset_path = ctx["files"]["dataset"]
        df = pd.read_csv(dataset_path) if dataset_path.endswith(".csv") else pd.read_excel(dataset_path)

        dataframe_info = self.get_dataframe_info(df)
        text_summary = self.analyze_dataframe(dataframe_info)

        ctx["findings"]["describe"] = dataframe_info
        ctx["insights"].append(text_summary)
        ctx["brief"] = "Dataset analyzed and described by DeepSeek."
        self.save_context(ctx)
        return ctx