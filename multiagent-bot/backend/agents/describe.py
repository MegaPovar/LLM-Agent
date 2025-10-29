import pandas as pd
import json
import requests
from io import StringIO

def get_dataframe_info(df):
    description = {
        "shape": df.shape,
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample_data": df.head(5).to_dict('records'),
    }
    return description

class DataFrameAnalyst:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "api_link"
    
    def analyze_dataframe(self, dataframe_info):
        
        # Формируем системный промпт с информацией о данных
        system_prompt = f"""
        Выдай одно предположение о чем датасет и напиши об этом текст. представь что ты описываешь
        его для начинающего дата аналитика.

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
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user"
            }
        ]
        
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload)
        return response.json()

analyst = DataFrameAnalyst(api_key)
result = analyst.analyze_dataframe(dataframe_info)
print(result['choices'][0]['message']['content'])
