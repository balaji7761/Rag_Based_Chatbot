import os
import json
import asyncio
from fastapi import FastAPI, Request, HTTPException
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import logging
from sse_starlette.sse import EventSourceResponse
from threading import Lock
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sqlite3
import os
from datetime import datetime
# Load environment variables
load_dotenv()
DB_PATH = "chatbot.db"
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chatbot_user_logins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                user_name TEXT,
                timestamp TEXT
            )
        ''')
        conn.commit()
 
init_db()
# Initialize FastAPI app
app = FastAPI(title="Passage Based Model API with RAG", version="1.2")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
MODEL = "phi3:medium"
llm = OllamaLLM(model=MODEL, system="You are an AI assistant for MPL. Answer only MPL-related questions based on the provided context.")

# Initialize embeddings (GPU enabled)
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="./model_all-MiniLM-L6-v2",
        model_kwargs={
            "local_files_only": True,
            "device": device
        }
    )
    logger.info(f"Successfully loaded local all-MiniLM-L6-v2 model on device: {device}")
except Exception as e:
    logger.error(f"Failed to load local model: {e}")
    raise RuntimeError("Could not load local embedding model.")

# Global variables
cache_file = "mplcache.json"
passage_file = "mplpassage.txt"
keywords_file = "mplkeys.txt"
vector_store_file = "faiss_index"
lock = Lock()
vectorizer = TfidfVectorizer()
vector_store = None
passage = ""
SYAT_KEYWORDS = set()
cache = {}

# Load passage and initialize vector store
def load_passage_and_vector_store():
    global passage, vector_store
    if os.path.exists(passage_file):
        with open(passage_file, "r", encoding="utf-8") as f:
            passage = f.read().strip()
        logger.info(f"Loaded passage: {passage[:100]}...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(passage)
        if os.path.exists(vector_store_file):
            try:
                vector_store = FAISS.load_local(vector_store_file, embeddings, allow_dangerous_deserialization=True)
                logger.info("Loaded existing vector store.")
            except Exception as e:
                logger.warning(f"Failed to load vector store: {e}. Creating new one.")
                vector_store = FAISS.from_texts(chunks, embeddings)
                vector_store.save_local(vector_store_file)
        else:
            vector_store = FAISS.from_texts(chunks, embeddings)
            vector_store.save_local(vector_store_file)
        logger.info(f"Vector store initialized with {len(chunks)} chunks.")
    else:
        passage = ""
        vector_store = None
        logger.warning("Passage file not found.")

# Load keywords
def load_keywords():
    global SYAT_KEYWORDS
    if os.path.exists(keywords_file):
        with open(keywords_file, "r", encoding="utf-8") as f:
            SYAT_KEYWORDS = set(f.read().strip().split("\n"))
        logger.info(f"Loaded {len(SYAT_KEYWORDS)} keywords.")
    else:
        SYAT_KEYWORDS = set()
        logger.warning("Keywords file not found.")

# Cache handling
def load_cache():
    if os.path.exists(cache_file):
        try:
            return json.load(open(cache_file, "r"))
        except json.JSONDecodeError:
            return {}
    return {}

def save_cache():
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=4)

# Init system
load_passage_and_vector_store()
load_keywords()
cache = load_cache()

# Keyword check
def is_related_to_syat(question: str) -> bool:
    return any(keyword.lower() in question.lower() for keyword in SYAT_KEYWORDS)

# TF-IDF cache search
def find_similar_question(question: str, threshold: float = 0.8):
    if not cache:
        return None
    questions = list(cache.keys()) + [question]
    tfidf_matrix = vectorizer.fit_transform(questions)
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    max_index = np.argmax(similarities)
    return questions[max_index] if similarities[max_index] >= threshold else None

# Prompt + QA chain
def create_qa_chain():
    prompt_template = """
You are an expert on the MPLD (Master Part List Development) system.
Answer the user's question strictly based on the context below.
Do NOT use any outside knowledge. If the context does not contain the answer,
respond with: "Ask related to the MPL."

Context:
{context}

Question: {question}
Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt | llm

# Main generator function
async def generate_response(question: str):
    if not passage or not vector_store:
        yield "Error: Passage or vector store not initialized."
        return
    if not is_related_to_syat(question):
        yield "I'm designed to handle MPL-related queries only. Ask anything related to MPL, and I'm here to assist."
        return
    if similar := find_similar_question(question):
        logger.info(f"Found similar question in cache: {similar}")
        yield cache[similar]
        return

    docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    #logger.info(f"Question: {question}")
    #logger.info(f"Retrieved context: {context}")

    qa_chain = create_qa_chain()
    response = qa_chain.invoke({"context": context, "question": question})

    #logger.info(f"Generated response: {response}")
    for chunk in response.split():
        yield chunk + " "
        await asyncio.sleep(0.1)
    cache[question] = response
    save_cache()

@app.get("/ask_question/")
async def ask_question(request: Request, question: str):
    async def event_generator():
        async for chunk in generate_response(question):
            if await request.is_disconnected():
                break
            yield {"data": chunk}
    return EventSourceResponse(event_generator())
@app.post("/log-user")
async def log_user(request: Request):
    try:
        body = await request.json()
        user_id = body.get("userId")
        user_name = body.get("userName")
        timestamp = datetime.now().isoformat()
 
        print("Logging:", user_id, user_name, timestamp)
 
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO chatbot_user_logins (user_id, user_name, timestamp) VALUES (?, ?, ?)",
                (user_id, user_name, timestamp)
            )
            conn.commit()
 
        return {"message": "User login recorded"}
 
    except Exception as e:
        return JSONResponse(content={"message": f"Error: {e}"}, status_code=500)
 
@app.get("/get-logins")
def get_logins():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM chatbot_user_logins ORDER BY timestamp DESC")
            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        return JSONResponse(content={"message": f"Error: {e}"}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="10.103.180.125",
        port=8056,
        ssl_certfile="mpl_cert.pem",
        ssl_keyfile="mpl_privkey.pem"
    )