# agents/rag.py
import os
import json
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from tavily import TavilyClient

from . import AgentBase


class RAGAgent(AgentBase):
    name = "rag"

    def __init__(
        self,
        apikey: str,
        baseurl: str = "https://api.deepseek.com/chat/completions",
        tavily_api_key: Optional[str] = None,
    ):
        self.apikey = apikey
        self.baseurl = baseurl

        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is not set in env or passed explicitly")

        self.tavily = TavilyClient(api_key=self.tavily_api_key)

    # ---------- ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ----------

    def extract_key_findings(self, ctx: Dict[str, Any]) -> List[str]:
        """Извлекает ключевые факты о датасете из describe + самого файла."""
        findings: List[str] = []

        # 1) Структура датасета из describe (если есть)
        describe_info = None
        if ctx.get("findings") and isinstance(ctx["findings"], dict):
            describe_info = ctx["findings"].get("describe")

        if isinstance(describe_info, dict):
            shape = describe_info.get("shape")
            if shape:
                findings.append(f"dataset shape: {shape}")
            num_cols = describe_info.get("numeric_columns") or describe_info.get("numericcolumns")
            cat_cols = describe_info.get("categorical_columns") or describe_info.get("categoricalcolumns")
            if num_cols:
                findings.append("numeric metrics: " + ", ".join(num_cols[:5]))
            if cat_cols:
                findings.append("categorical dimensions: " + ", ".join(cat_cols[:5]))

        # 2) Если describe ничего не записал — читаем файл напрямую
        dataset_path = ctx["files"]["dataset"]
        df = pd.read_csv(dataset_path) if dataset_path.endswith(".csv") else pd.read_excel(dataset_path)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()

        if not any("numeric metrics:" in f for f in findings) and numeric_cols:
            findings.append("numeric metrics: " + ", ".join(numeric_cols[:5]))
        if not any("categorical dimensions:" in f for f in findings) and cat_cols:
            findings.append("categorical dimensions: " + ", ".join(cat_cols[:5]))

        # 3) Пользовательский промпт (бизнес-контекст)
        if ctx.get("prompt"):
            findings.append("business goal: " + ctx["prompt"][:160])

        return findings[:6]

    def generate_search_queries(self, findings: List[str]) -> List[str]:
        """Генерирует 1–3 поисковых запроса для Tavily на основе структуры датасета и цели."""
        queries: List[str] = []

        for f in findings:
            if "numeric metrics:" in f:
                queries.append("KPI benchmarks for " + f.replace("numeric metrics:", ""))
            elif "categorical dimensions:" in f:
                queries.append("segmentation best practices for " + f.replace("categorical dimensions:", ""))
            elif "business goal:" in f:
                queries.append("data-driven strategies " + f.replace("business goal:", ""))

            if len(queries) >= 3:
                break

        if not queries:
            queries = ["best practices for analysing business tabular datasets"]

        return queries[:3]

    def search_and_retrieve(self, queries: List[str]) -> str:
        """Реальный поисковый запрос к Tavily и агрегация контекста."""
        contexts: List[str] = []

        for q in queries:
            try:
                ctx = self.tavily.get_search_context(
                    query=q,
                    search_depth="advanced",
                    max_tokens=800,
                )
                contexts.append(f"### Query: {q}\n{ctx}")
            except Exception as e:
                contexts.append(f"### Query: {q}\nTavily error: {e}")

        return "\n\n---\n\n".join(contexts) if contexts else "No external context fetched."

    def build_dataset_summary(self, ctx: Dict[str, Any]) -> str:
        """Использует описание датасета из DescribeAgent + базовую info по датафрейму."""
        parts: List[str] = []

        # 1) HTML/текст из describe (если есть)
        describe_text = None
        if isinstance(ctx.get("insights"), list):
            # Берём первое insight от describe, если он там
            for msg in ctx["insights"]:
                if isinstance(msg, str) and "Telegram-HTML" in msg:
                    describe_text = msg
                    break

        if describe_text:
            parts.append("Dataset description (from DescribeAgent):")
            parts.append(describe_text)

        # 2) Мини‑саммари из файла
        dataset_path = ctx["files"]["dataset"]
        df = pd.read_csv(dataset_path) if dataset_path.endswith(".csv") else pd.read_excel(dataset_path)

        parts.append(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        parts.append("Columns: " + ", ".join(df.columns.tolist()[:15]))

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            parts.append("Example numeric columns: " + ", ".join(num_cols[:5]))
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        if cat_cols:
            parts.append("Example categorical columns: " + ", ".join(cat_cols[:5]))

        return "\n".join(parts)

    def generate_insights_with_rag(
        self,
        dataset_summary: str,
        web_context: str,
        user_prompt: str,
    ) -> str:
        """RAG-вызов DeepSeek: интерпретация датасета + внешние знания."""
        system_prompt = """
Ты аналитик данных и консультант по BI.
У тебя есть:
1) Описание табличного датасета (структура, колонки, типы, базовые характеристики).
2) Консолидированный контекст из Интернета (кейсы, бенчмарки, best practices).

Требования к ответу:
- Опираться на структуру и смысл колонок, делать разумные гипотезы о связях,
  но не выдавать их как строго доказанные.
- Использовать внешние источники как примеры, типичные паттерны и бенчмарки.
- Формат ответа — JSON:

{
  "key_insights": [...],
  "potential_relationships": [...],
  "benchmarks": [...],
  "recommendations": [...],
  "risks_and_caveats": [...],
  "html_summary": "<p>...</p>"
}

Пиши по-русски, максимально конкретно.
"""

        user_content = f"""
ПОЛЬЗОВАТЕЛЬСКИЙ ПРОМПТ (цель анализа):
{user_prompt}

ОПИСАНИЕ ДАТАСЕТА:
{dataset_summary}

ВНЕШНИЕ ИСТОЧНИКИ (Tavily):
{web_context}
"""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.apikey}",
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_content.strip()},
            ],
            "temperature": 0.25,
            "max_tokens": 1800,
        }

        try:
            r = requests.post(self.baseurl, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return json.dumps(
                {
                    "key_insights": [],
                    "potential_relationships": [],
                    "benchmarks": [],
                    "recommendations": [],
                    "risks_and_caveats": [f"Ошибка генерации RAG-инсайтов: {e}"],
                    "html_summary": "<p>RAG-анализ не удалось выполнить из-за ошибки API.</p>",
                },
                ensure_ascii=False,
                indent=2,
            )

    # ---------- ОСНОВНОЙ МЕТОД АГЕНТА ----------

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        print("🧠 RAGAgent: starting external research with Tavily...")

        # 1. Извлекаем находки и делаем запросы
        findings = self.extract_key_findings(ctx)
        queries = self.generate_search_queries(findings)
        print(f"[RAG] Search queries: {queries}")

        # 2. Реальный веб-поиск
        web_context = self.search_and_retrieve(queries)

        ctx.setdefault("external_research", {})
        ctx["external_research"]["search_queries"] = queries
        ctx["external_research"]["web_context"] = web_context

        # 3. «Саммари» датасета (без StatAgent)
        dataset_summary = self.build_dataset_summary(ctx)

        # 4. RAG-вызов DeepSeek
        rag_json = self.generate_insights_with_rag(
            dataset_summary=dataset_summary,
            web_context=web_context,
            user_prompt=ctx.get("prompt", ""),
        )

        # 5. Сохраняем в контекст
        ctx.setdefault("findings", {})
        ctx.setdefault("insights", [])
        ctx.setdefault("metrics", {})

        ctx["findings"]["rag_raw"] = rag_json
        ctx["insights"].append(
            "RAG insights generated from dataset description and external web context."
        )
        ctx["metrics"]["rag_used"] = True

        # 6. Пишем отчёт
        # outdir = Path(ctx["files"]["outdir"])
        files_info = ctx.get("files", {})
        outdir_str = files_info.get("outdir")
        if outdir_str is None:
            # фоллбек: использовать каталог, где лежит датасет
            dataset_path = Path(files_info["dataset"])
            outdir = dataset_path.parent
        else:
            outdir = Path(outdir_str)
        outdir.mkdir(parents=True, exist_ok=True)
        report_path = outdir / "rag_report.json"
        report_path.write_text(rag_json, encoding="utf-8")

        ctx["files"].setdefault("artifacts", [])
        ctx["files"]["artifacts"].append(str(report_path))
        ctx["brief"] = "RAG: внешние бенчмарки и рекомендации на основе описания датасета."

        self.save_context(ctx)
        print("✅ RAGAgent: external research completed.")
        return ctx