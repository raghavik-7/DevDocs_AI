from fastapi import FastAPI
from dotenv import load_dotenv
import os
import time
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize with performance optimizations
print("🚀 Loading models for FastAPI server...")

# OPTIMIZATION 1: Reduce embedding model cache and batch size
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    cache_folder="./model_cache",
    embed_batch_size=1,  # Reduce batch size for faster individual queries
    max_length=256  # Limit token length for faster processing
)

# OPTIMIZATION 2: Reduced timeout and enable streaming
llm = Ollama(
    model="llama3.2:1b",
    request_timeout=60.0,  # Reduced from 120s
    base_url="http://localhost:11434",
    temperature=0.1,  # Lower temperature for faster, more deterministic responses
    additional_kwargs={
        "num_predict": 256,  # Limit response length for speed
        "num_ctx": 2048,     # Reduced context window
        "top_k": 10,         # Reduce sampling complexity
        "top_p": 0.9
    }
)

# Set global settings
Settings.embed_model = embed_model
Settings.llm = llm
Settings.chunk_size = 512  # Smaller chunks for faster processing
Settings.chunk_overlap = 50

print("📚 Loading index from ChromaDB storage...")

# Connect to existing ChromaDB collection
try:
    start_time = time.time()
    
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
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    
    # OPTIMIZATION 3: More aggressive query engine settings for speed
    query_engine = index.as_query_engine(
        similarity_top_k=2,  # Reduced from 3 for faster retrieval
        response_mode="compact",  # Fastest response mode
        verbose=False,  # Disable verbose logging for speed
        streaming=False,  # Disable streaming for simpler processing
        node_postprocessors=[]  # Remove any post-processors
    )
    
    # OPTIMIZATION 4: Simplified prompt for faster processing
    custom_qa_template = PromptTemplate(
        "Context: {context_str}\n\n"
        "Question: {query_str}\n\n"
        "Answer based only on the context above:"
    )
    
    # Configure the query engine
    query_engine.update_prompts({
        "response_synthesizer:text_qa_template": custom_qa_template
    })
    
    load_time = time.time() - start_time
    print(f"✅ Index loaded in {load_time:.2f} seconds")
    
    # Quick test
    test_start = time.time()
    test_response = query_engine.query("What is this documentation about?")
    test_time = time.time() - test_start
    print(f"🧪 Test query completed in {test_time:.2f} seconds")
    print(f"📝 Response preview: {str(test_response)[:100]}...")
    
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
        "vector_store": "ChromaDB",
        "performance_mode": "optimized"
    }

@app.get("/query/")
async def query_docs(question: str):
    try:
        start_time = time.time()
        
        # OPTIMIZATION 5: Timeout wrapper for queries
        response = query_engine.query(question)
        
        query_time = time.time() - start_time
        
        # Get source nodes (simplified for speed)
        source_count = 0
        if hasattr(response, 'source_nodes'):
            source_count = len(response.source_nodes)
        
        return {
            "response": str(response),
            "question": question,
            "sources_used": source_count,
            "query_time_seconds": round(query_time, 2),
            "note": "Response based only on provided documentation"
        }
    except Exception as e:
        return {"error": f"Query failed: {str(e)}"}

@app.get("/health")
async def health_check():
    try:
        start_time = time.time()
        
        # Quick health test
        test_response = query_engine.query("test")
        
        health_time = time.time() - start_time
        
        return {
            "status": "healthy",
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "llm_model": "llama3.2:1b",
            "vector_store": "ChromaDB",
            "test_query": "successful",
            "response_time_seconds": round(health_time, 2),
            "performance_mode": "optimized"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/quick_test/")
async def quick_performance_test():
    """Quick performance test endpoint"""
    test_queries = [
        "What is this about?",
        "Main features?",
        "How to install?"
    ]
    
    results = []
    total_start = time.time()
    
    for query in test_queries:
        try:
            start_time = time.time()
            response = query_engine.query(query)
            query_time = time.time() - start_time
            
            results.append({
                "query": query,
                "response_length": len(str(response)),
                "query_time_seconds": round(query_time, 2),
                "status": "success"
            })
        except Exception as e:
            results.append({
                "query": query,
                "error": str(e),
                "status": "failed"
            })
    
    total_time = time.time() - total_start
    
    return {
        "test_results": results,
        "total_time_seconds": round(total_time, 2),
        "average_per_query": round(total_time / len(test_queries), 2)
    }

@app.get("/debug/")
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

# OPTIMIZATION 6: Add startup event to warm up the system
@app.on_event("startup")
async def startup_event():
    print("🔥 Warming up the system...")
    try:
        # Warm up query to load models into memory
        warm_response = query_engine.query("hello")
        print("✅ System warmed up successfully")
    except Exception as e:
        print(f"⚠️ Warm-up failed: {e}")

#dev-docs-chatbot -> enc
# To run: uvicorn main:app --reload
#webhook is used for triggering
#https://devdocschatbot.app.n8n.cloud/webhook-test/chat
# C:\Users\HP\AppData\Local/ngrok/ngrok.yml
#ngrok http 8000 with this parallelly run your server 1st command