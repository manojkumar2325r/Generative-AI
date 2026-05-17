from fastapi import FastAPI
from pydantic import BaseModel
from rag_bot import ask_cricket_bot

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    print("Received:", query.question)  # 👈 ADD THIS

    answer, sources = ask_cricket_bot(query.question)

    return {
        "answer": answer,
        "sources": list(sources)
    }