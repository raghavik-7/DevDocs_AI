# Developer Documentation Chatbot (AI Chatbot for Developer Documentation Assistance)

## 🚀 Overview
This project is an AI-powered chatbot designed to assist developers by answering questions about documentation. It leverages FastAPI for the backend, ChromaDB for vector storage, HuggingFace embeddings for semantic search, and Ollama for LLM-based responses. The system is optimized for fast, efficient, and contextually relevant answers based solely on your provided documentation.

## ✨ Features
- **Semantic Search**: Uses state-of-the-art embeddings (BAAI/bge-small-en-v1.5) for document retrieval.
- **LLM-Powered Answers**: Utilizes Ollama (llama3.2:1b) for generating concise, context-aware answers.
- **ChromaDB Integration**: Stores and retrieves document vectors for efficient similarity search.
- **Performance Optimizations**: Multiple optimizations for fast response times and low resource usage.
- **REST API**: Exposes endpoints for querying, health checks, debugging, and performance testing.
- **Easy Deployment**: Run locally with Uvicorn and expose via ngrok for webhook integrations.

## 🏗️ Architecture
```
User Query → FastAPI → ChromaDB (Vector Search) → LLM (Ollama) → Response
```
- **FastAPI**: Handles HTTP requests and API endpoints.
- **ChromaDB**: Stores vectorized documentation for semantic search.
- **HuggingFace Embeddings**: Converts text to vectors for similarity search.
- **Ollama LLM**: Generates answers based on retrieved context.

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone <repo-url>
cd AI_chatbot_for_developer_documentation_assistance
```

### 2. Set Up Python Environment
It's recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*If `requirements.txt` is missing, install the following main packages:*
- fastapi
- uvicorn
- python-dotenv
- llama-index
- chromadb
- huggingface_hub

### 4. Prepare Model Cache and Storage
- Ensure the `model_cache/` and `storage/` directories exist.
- Place your documentation files in the appropriate location and index them as needed (see below).

### 5. Environment Variables
Create a `.env` file if you need to set environment variables (optional).

### 6. Start the FastAPI Server
```bash
uvicorn main:app --reload
```

### 7. (Optional) Expose Locally with ngrok
To make your API accessible externally (e.g., for webhooks):
```bash
ngrok http 8000
```

## 🧑‍💻 Usage

### API Endpoints
- `GET /` — Root endpoint, returns status and info.
- `GET /query/?question=...` — Query the documentation with a question.
- `GET /health` — Health check for the system.
- `GET /quick_test/` — Runs a set of test queries for performance benchmarking.
- `GET /debug/` — Returns debug info about storage and ChromaDB collections.

#### Example Query
```bash
curl "http://localhost:8000/query/?question=What%20is%20this%20documentation%20about?"
```

#### Example Response
```json
{
  "response": "This documentation covers ...",
  "question": "What is this documentation about?",
  "sources_used": 2,
  "query_time_seconds": 0.45,
  "note": "Response based only on provided documentation"
}
```

## ⚡ Performance Optimizations
- **Reduced Embedding Batch Size**: Faster individual queries.
- **Shorter LLM Timeouts**: Quicker error handling and response.
- **Limited Response Length**: Faster LLM inference.
- **Smaller Chunks & Overlap**: Optimized for retrieval speed.
- **Aggressive Query Engine Settings**: Lower `similarity_top_k`, compact response mode, no post-processing.
- **Prompt Simplification**: Minimal prompt for faster LLM response.
- **Startup Warmup**: Loads models into memory at server start.

## 🛠️ Troubleshooting & Debugging
- Use `/debug/` to inspect storage and ChromaDB collections.
- Use `/health` to check if the system is healthy and models are loaded.
- Check server logs for detailed error messages.
- If ChromaDB errors occur, ensure the `storage/` directory is present and collections are properly indexed.

## 🤝 Contributing
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and add tests if applicable.
4. Submit a pull request with a clear description of your changes.

---

**Contact:** For questions or support, open an issue or contact the maintainer. 
