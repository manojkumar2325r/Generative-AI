from dotenv import load_dotenv
import os

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# Load vector store
print("🏏 Loading cricket knowledge base...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

# Chat history
chat_history = []

# Main function
def ask_cricket_bot(question):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = set(doc.metadata.get("source", "unknown") for doc in docs)

    messages = [
        {
            "role": "system",
            "content": """You are an expert cricket analyst.
Answer using the context. If not found, say clearly.

Context:
""" + context
        }
    ]

    for msg in chat_history:
        messages.append(msg)

    messages.append({"role": "user", "content": question})

    response = llm.invoke(messages)
    answer = response.content

    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    return answer, sources