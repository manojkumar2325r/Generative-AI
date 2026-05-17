import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

print("✅ Imports loaded!")

# ── 1. Load PDFs ───────────────────────────────────────────────
print("📄 Loading cricket PDFs...")
pdf_loader = PyPDFDirectoryLoader("./docs")
pdf_docs = pdf_loader.load()
print(f"   ✅ {len(pdf_docs)} PDF pages loaded")

# ── 2. Load Websites ───────────────────────────────────────────
print("🌐 Loading cricket websites...")
urls = [
    "https://www.icc-cricket.com/about/cricket/history-of-cricket",
    "https://www.lords.org/mcc/the-laws-of-cricket",
]
try:
    web_loader = WebBaseLoader(urls)
    web_docs = web_loader.load()
    print(f"   ✅ {len(web_docs)} website pages loaded")
except Exception as e:
    print(f"   ⚠️  Website loading failed: {e}")
    web_docs = []

# ── 3. Load Wikipedia ──────────────────────────────────────────
print("📖 Loading Wikipedia pages...")
wikipedia_topics = [
    "Cricket",
    "Test cricket",
    "One Day International",
    "Twenty20 cricket",
    "Indian Premier League",
    "ICC Cricket World Cup",
    "Sachin Tendulkar",
    "Virat Kohli",
    "MS Dhoni",
    "Ricky Ponting",
    "Brian Lara",
    "Shane Warne",
    "History of cricket",
    "Duckworth-Lewis-Stern method",
    "Decision Review System",
]
wiki_docs = []
for topic in wikipedia_topics:
    try:
        loader = WikipediaLoader(
            query=topic,
            load_max_docs=1,
            doc_content_chars_max=4000,
        )
        docs = loader.load()
        wiki_docs.extend(docs)
        print(f"   ✅ Loaded: {topic}")
    except Exception as e:
        print(f"   ⚠️  Skipped {topic}: {e}")
print(f"   ✅ {len(wiki_docs)} Wikipedia pages loaded")

# ── 4. Combine + Chunk ─────────────────────────────────────────
print("✂️  Chunking documents...")
all_docs = pdf_docs + web_docs + wiki_docs
print(f"   📦 Total documents: {len(all_docs)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "],
)
chunks = splitter.split_documents(all_docs)
print(f"   ✅ {len(chunks)} chunks created")

# ── 5. Embed + Store ───────────────────────────────────────────
print("🔢 Embedding using free HuggingFace model...")
print("   (downloading model first time, please wait 1-2 mins...)")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)
print("✅ Cricket knowledge base ready!")
print(f"✅ Total vectors stored: {len(chunks)}")