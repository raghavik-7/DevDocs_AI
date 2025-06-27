from fastapi import FastAPI
from dotenv import load_dotenv
import os
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize the same models and settings as in app2.py
print("🚀 Loading models for FastAPI server...")

# Load the same embedding model
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    cache_folder="./model_cache"
)

# Set up Ollama LLM (use the same model you used in app2.py)
llm = Ollama(
    model="llama3.2:1b",  # Change this to match the model you used in app2.py
    request_timeout=120.0,
    base_url="http://localhost:11434"
)

# Set global settings
Settings.embed_model = embed_model
Settings.llm = llm

print("📚 Loading index from ChromaDB storage...")

# Connect to existing ChromaDB collection
try:
    chroma_client = chromadb.PersistentClient(path="./storage")
    collection_name = 'dev_docs_collection'
    
    # Get the existing collection
    chroma_collection = chroma_client.get_collection(collection_name)
    print(f"✅ Found ChromaDB collection with {chroma_collection.count()} documents")
    
    # Create vector store from existing collection
    vector_store = ChromaVectorStore(chroma_collection)
    
    # Create storage context with the vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Recreate the index from the existing vector store
    # This is the key: we recreate the index from the vector store, not load from storage
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    
    # Create query engine
    query_engine = index.as_query_engine()
    
    print("✅ Index reconstructed successfully from ChromaDB!")
    
    # Test the setup
    test_response = query_engine.query("What is this documentation about?")
    print(f"🧪 Test query successful: {str(test_response)[:100]}...")
    
except Exception as e:
    print(f"❌ Error loading ChromaDB index: {e}")
    print("🔧 DEBUGGING INFO:")
    
    try:
        chroma_client = chromadb.PersistentClient(path="./storage")
        collections = chroma_client.list_collections()
        print(f"   Available collections: {[c.name for c in collections]}")
        
        if collections:
            for collection in collections:
                print(f"   Collection '{collection.name}': {collection.count()} items")
    except Exception as debug_e:
        print(f"   ChromaDB debug error: {debug_e}")
    
    raise e

@app.get("/")
async def root():
    return {
        "message": "Developer documentation chatbot is running with ChromaDB",
        "status": "loaded",
        "vector_store": "ChromaDB"
    }

@app.get("/query/")
async def query_docs(question: str):
    try:
        response = query_engine.query(question)
        return {
            "response": str(response),
            "question": question
        }
    except Exception as e:
        return {"error": f"Query failed: {str(e)}"}

@app.get("/health")
async def health_check():
    try:
        # Test query to ensure everything is working
        test_response = query_engine.query("test")
        return {
            "status": "healthy",
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "llm_model": "llama3.2:1b",
            "vector_store": "ChromaDB",
            "test_query": "successful"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/debug")
async def debug_storage():
    """Debug endpoint to check storage status"""
    debug_info = {
        "storage_directory_exists": os.path.exists('./storage'),
        "storage_contents": []
    }
    
    if os.path.exists('./storage'):
        debug_info["storage_contents"] = os.listdir('./storage')
    
    try:
        chroma_client = chromadb.PersistentClient(path="./storage")
        collections = chroma_client.list_collections()
        debug_info["chromadb_collections"] = []
        
        for collection in collections:
            debug_info["chromadb_collections"].append({
                "name": collection.name,
                "count": collection.count()
            })
            
    except Exception as e:
        debug_info["chromadb_error"] = str(e)
    
    return debug_info

# To run: uvicorn main:app --reload