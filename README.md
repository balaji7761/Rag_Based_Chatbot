# 🤖 RAG + Agent-based Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot enhanced with an **agent layer** to handle both knowledge-based queries and task automation.  
This project demonstrates **LLM-powered question answering, intelligent agent orchestration, and production-ready deployment**.

---

## 🚀 Features

- **RAG Pipeline**
  - Uses FAISS as a vector store with HuggingFace embeddings.
  - Caches responses with TF-IDF to reduce redundant calls.
  - Provides accurate document-based Q&A.

- **Agent Integration**
  - Detects user intent (e.g., *raise a new request*).
  - Executes actions via REST API calls while still providing RAG answers.
  - Example: Submitting an **MPL Access Request** via external API.

- **Backend**
  - Built with **FastAPI** for lightweight, high-performance APIs.
  - Async endpoints for improved throughput.

- **LLM**
  - Powered by **Ollama (Phi-3)** for low-latency responses.
  - Flexible to integrate other open-source or cloud LLMs.

- **Deployment Ready**
  - Multi-user support.
  - Can be containerized with Docker.
  - Production-tested.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    A[User Query] --> B[FastAPI Backend]
    B --> C[RAG Layer: Embeddings + FAISS + TF-IDF Cache]
    B --> D[Agent Layer: Intent Detection + REST API Calls]
    C --> E[LLM (Ollama - Phi-3)]
    D --> E
    E --> F[Response to User]
