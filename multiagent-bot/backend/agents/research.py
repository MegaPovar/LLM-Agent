from ddgs import DDGS
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def web_search_and_filter(query: str, max_results=5, sim_threshold=0.4):
    with DDGS() as ddgs:
        snippets = []
        for r in ddgs.text(query, max_results=max_results):
            snippets.append(r["body"])
    if not snippets:
        return []

    q_emb = model.encode([query])
    snippets_emb = model.encode(snippets)
    similarities = cosine_similarity(q_emb, snippets_emb)[0]
    filtered = [snippet for snippet, sim in zip(snippets, similarities) if sim >= sim_threshold]
    return filtered

def llm_generate(query, relevant_snippets):
    if not relevant_snippets:
        return f"По запросу '{query}' не найдено технически релевантных данных."
    context = "\n".join(relevant_snippets)
    return f"Вот релевантная информация из интернета по запросу '{query}':\n{context}"

