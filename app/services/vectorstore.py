from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

dimension = 768
index = faiss.IndexFlatL2(dimension)

documents = []


def chunk_text(text, chunk_size=300, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def add_to_vectorstore(text):
    chunks = chunk_text(text)
    if not chunks:
        return
    vectors = model.encode(chunks)
    index.add(np.array(vectors, dtype=np.float32))
    documents.extend(chunks)


def keyword_search(query, top_n=5):
    """Fallback: return chunks that contain query keywords."""
    query_words = set(query.lower().split())
    scored = []
    for i, doc in enumerate(documents):
        doc_lower = doc.lower()
        score = sum(1 for word in query_words if word in doc_lower)
        if score > 0:
            scored.append((score, i, doc))
    scored.sort(reverse=True)
    return [doc for _, _, doc in scored[:top_n]]


def search(query, k=10):
    if index.ntotal == 0:
        return []

    actual_k = min(k, index.ntotal)
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec, dtype=np.float32), actual_k)

    semantic_results = []
    for i in indices[0]:
        if i != -1 and i < len(documents):
            semantic_results.append(documents[i])

    keyword_results = keyword_search(query, top_n=5)

    seen = set()
    combined = []
    for doc in semantic_results + keyword_results:
        if doc not in seen:
            seen.add(doc)
            combined.append(doc)

    return combined[:10]
