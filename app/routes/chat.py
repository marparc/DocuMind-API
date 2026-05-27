from fastapi import APIRouter
from pydantic import BaseModel

from app.services.vectorstore import search
from app.services.llm import generate_answer

router = APIRouter()


class Query(BaseModel):
    question: str


@router.post("/chat")
def chat(query: Query):
    docs = search(query.question)

    if not docs:
        return {
            "answer": "No documents found. Please upload a file first.",
            "sources": [],
        }

    print("=== RETRIEVED CHUNKS ===")
    for i, doc in enumerate(docs):
        print(f"[{i}] {doc[:300]}\n")
    print("========================")

    context = "\n\n".join(docs[:8])

    answer = generate_answer(query.question, context)

    return {"answer": answer, "sources": docs[:8]}
