# import json
# import pandas as pd
# import numpy as np
# from scipy.stats import pearsonr
# import sys
# import ollama
# OLLAMA_AVAILABLE = True


# class TrivialityAgentWithLLM:
#     def init(self, df, metadata=None, llm_model="llama3.2"):
#         self.df = df
#         self.metadata = metadata or {}
#         self.llm_model = llm_model
#         self.correlations = {}
#         self.triviality_scores = {}

#     def detect_correlations(self, method="pearson", threshold=0.5):
#         numeric_cols = self.df.select_dtypes(include=[np.number]).columns
#         for i, col1 in enumerate(numeric_cols):
#             for col2 in numeric_cols[i + 1:]:
#                 try:
#                     x, y = self.df[col1].dropna(), self.df[col2].dropna()
#                     # Выравниваем индексы
#                     common_idx = x.index.intersection(y.index)
#                     if len(common_idx) < 10:
#                         continue
#                     corr, _ = pearsonr(x[common_idx], y[common_idx])
#                     if abs(corr) >= threshold:
#                         self.correlations[(col1, col2)] = float(corr)
#                 except Exception as e:
#                     print(f"Ошибка при вычислении корреляции {col1}-{col2}: {e}")
#         return self.correlations

#     def query_llm_for_triviality(self, col1, col2):
#         if not OLLAMA_AVAILABLE:
#             return {
#                 "is_trivial": False,
#                 "reason": "LLM недоступен — используется эвристика",
#                 "common_cause": None,
#                 "confidence": 0.5
#             }

#         desc1 = self.metadata.get(col1, "")
#         desc2 = self.metadata.get(col2, "")

#         prompt = f"""You are an expert data scientist. Analyze whether the correlation between two variables is likely to be trivial or meaningful. Check if words from variables are
#         often used in shared context, but also use more usual way to check triviality. Check if variables use two or more common words. It may mean they are trivial, but
#         only if words are rare ones and not general, like 'annual', 'count', etc
# Variable 1: "{col1}"{f' (description: "{desc1}")' if desc1 else ''}
# Variable 2: "{col2}"{f' (description: "{desc2}")' if desc2 else ''}

# Answer in strict JSON format with the following keys:
# - "is_trivial": boolean
# - "reason": string
# - "common_cause": string or null
# - "confidence": float (0.0 to 1.0)

# Output only valid JSON. No extra text."""

#         try:
#             response = ollama.chat(
#                 model=self.llm_model,
#                 messages=[{"role": "user", "content": prompt}],
#                 options={"temperature": 0.2} #it was 0.1
#             )
#             raw_text = response["message"]["content"].strip()

#             # Иногда LLM добавляет "json", уберём это
#             if raw_text.startswith("```json"):
#                 raw_text = raw_text[7:]
#             if raw_text.endswith(""):
#                 raw_text = raw_text[:-3]

#             result = json.loads(raw_text)
#             # Валидация
#             assert isinstance(result.get("is_trivial"), bool)
#             assert isinstance(result.get("reason"), str)
#             assert isinstance(result.get("confidence"), (int, float))
#             result["confidence"] = float(result["confidence"])
#             return result
#         except Exception as e:
#             print(f"⚠️ Ошибка LLM для {col1}-{col2}: {e}")
#             return {
#                 "is_trivial": False,
#                 "reason": f"Ошибка LLM: {str(e)[:50]}",
#                 "common_cause": None,
#                 "confidence": 0.5
#             }

#     def evaluate_triviality_with_llm(self):
#         for (col1, col2), corr in self.correlations.items():
#             print(f"Анализирую: {col1} ↔️ {col2} (r={corr:.3f})")

#             llm_result = self.query_llm_for_triviality(col1, col2)
#             score = llm_result["confidence"]


# self.triviality_scores[(col1, col2)] = {
#                 "correlation": corr,
#                 "triviality_score": np.clip(score, 0.0, 1.0),
#                 "is_trivial": llm_result["is_trivial"],
#                 "reasons": [llm_result["reason"]],
#                 "common_cause": llm_result.get("common_cause"),
#                 "source": "llm"
#             }
#         return self.triviality_scores

#     def report(self):
#         print("\n" + "=" * 60)
#         print("ОТЧЁТ АГЕНТА С ИСПОЛЬЗОВАНИЕМ LLM")
#         print("=" * 60)
#         if not self.triviality_scores:
#             print("Нет корреляций, превышающих порог.")
#             return
#         for (col1, col2), info in self.triviality_scores.items():
#             status = "⚠️ ТРИВИАЛЬНО" if (info['triviality_score'] > 0.5) else "✅ НЕТРИВИАЛЬНО"
#             print(f"\n{status}: '{col1}' ↔️ '{col2}'")
#             print(f"  Корреляция: {info['correlation']:.3f}")
#             print(f"  Уверенность в тривиальности: {info['triviality_score']:.2f}")
#             if info['triviality_score'] > 0.5:
#                 print(f"  Причина: {info['reasons'][0]}")
#                 if info["common_cause"]:
#                     print(f"  Общая причина: {info['common_cause']}")
#             print(f"  Источник: {info['source']}")


# # === ТЕСТОВЫЙ ЗАПУСК ===
# if name == "main":
#     np.random.seed(42)
#     n = 500
#     temperature = np.random.uniform(20, 35, n)
#     ice_cream_sales = temperature * 8 + np.random.normal(0, 5, n)
#     drowning_incidents = temperature * 0.1 + np.random.normal(0, 1, n)
#     ice_cream_shops_revenue = temperature * 0.5 + np.random.normal(0, 5, n)
#     book_sales = np.random.normal(100, 10, n)

#     df = pd.DataFrame({
#         "ice_cream_sales": ice_cream_sales,
#         "drowning_incidents": drowning_incidents,
#         "ice_cream_shops_revenue": ice_cream_shops_revenue,
#         "book_sales": book_sales
#     })

#     metadata = {
#         "ice_cream_sales": "Daily units sold",
#         "drowning_incidents": "Daily drowning cases",
#         "book_sales": "Number of books sold per day",
#         "ice_cream_shops_revenue":"Daily revenue in tens thousands rubles"
#     }
#     print("🔍 Обнаружение корреляций...")
#     agent = TrivialityAgentWithLLM(df, metadata=metadata, llm_model="llama3.2")
#     agent.detect_correlations(threshold=0.4)
#     print(f"Найдено корреляций: {len(agent.correlations)}")

#     print("\n🧠 Оценка тривиальности с помощью LLM...")
#     agent.evaluate_triviality_with_llm()
#     agent.report()
from . import AgentBase

class EvalAgent(AgentBase):
    name = "eval"
    def run(self, ctx):
        ctx["brief"] = "Evaluation complete."
        self.save_context(ctx)
        return ctx