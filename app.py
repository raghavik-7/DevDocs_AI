import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
import requests

# Load environment variables
load_dotenv()

print("🚀 Starting FREE Local RAG Setup...")
print(f"📁 Project Location: {os.getcwd()}")

# Function to check available Ollama models
def check_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            available_models = [model['name'] for model in models.get('models', [])]
            print(f"📋 Available Ollama models: {available_models}")
            return available_models
        else:
            print("❌ Cannot connect to Ollama service")
            return []
    except Exception as e:
        print(f"❌ Error checking Ollama models: {e}")
        return []

# Check what models are available
available_models = check_ollama_models()

# Use free local embedding model
print("Loading embedding model (this may take a moment on first run)...")
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",  # Fast, good quality, free
    cache_folder="./model_cache"  # Cache on D: drive in your project
)

# Choose the best available model based on what's installed
print("Setting up connection to Ollama (on C: drive)...")
model_to_use = None

# Priority order: smallest to largest
model_priority = [
    "llama3.2:1b",
    "phi3:mini", 
    "llama3.2:3b-instruct-q4_0",
    "llama3.2:3b-instruct-q4_K_M",
    "gemma2:2b",
    "qwen2:1.5b"
]

for model in model_priority:
    if model in available_models:
        model_to_use = model
        break

if not model_to_use:
    # If no preferred models found, use the first available model
    if available_models:
        model_to_use = available_models[0]
        print(f"⚠️  Using first available model: {model_to_use}")
    else:
        print("❌ No models found. Please install a model first:")
        print("   Run: ollama pull llama3.2:1b")
        print("   Then run this script again.")
        exit(1)

try:
    llm = Ollama(
        model=model_to_use,
        request_timeout=120.0,
        base_url="http://localhost:11434"
    )
    print(f"✅ Ollama connection configured with model: {model_to_use}")
except Exception as e:
    print(f"❌ Ollama setup error: {e}")
    print("💡 Suggested commands:")
    print("   ollama pull llama3.2:1b")
    print("   ollama serve")
    exit(1)

# Set global models
Settings.embed_model = embed_model
Settings.llm = llm

print("Loading documents from ./documentation...")
try:
    documents = SimpleDirectoryReader('./documentation').load_data()
    print(f"📚 Loaded {len(documents)} documents")
except Exception as e:
    print(f"❌ Error loading documents: {e}")
    print("💡 Make sure ./documentation folder exists with your docs")
    exit(1)

# Setup ChromaDB (storage on D: drive)
print("Setting up vector database...")
chroma_client = chromadb.PersistentClient(path="./storage")
collection_name = 'dev_docs_collection'

try:
    # Clear existing collection if it exists
    try:
        chroma_client.delete_collection(collection_name)
        print("🧹 Cleared existing collection")
    except:
        pass
    chroma_collection = chroma_client.create_collection(collection_name)
    print("✅ Created new collection")
except Exception as e:
    print(f"Collection setup error: {e}")
    chroma_collection = chroma_client.create_collection(collection_name)

vector_store = ChromaVectorStore(chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Creating embeddings (local processing, no API calls)...")
print("Progress: ", end="", flush=True)

# Process in batches for progress tracking
batch_size = 20
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    
    if i == 0:
        # Create index with first batch
        index = VectorStoreIndex.from_documents(
            batch, 
            storage_context=storage_context,
            show_progress=True
        )
    else:
        # Add remaining documents
        for doc in batch:
            index.insert(doc)
    
    # Show progress
    progress = min(i + batch_size, len(documents))
    print(f"█", end="", flush=True)

print(f"\n\n✅ Embedding creation complete!")

# Persist the index (saved on D: drive)
index.storage_context.persist('./storage')
print("💾 Index saved to D: drive ./storage directory")

print("\n📊 SUMMARY:")
print(f"   📁 Documents processed: {len(documents)}")
print(f"   💾 Embeddings stored: D:/storage")
print(f"   🤖 LLM service: C:/Ollama (via HTTP)")
print(f"   🆓 Total API costs: $0.00")
print(f"   📱 Model used: {model_to_use}")

# Test the complete setup
print("\n🧪 Testing the complete system...")
try:
    query_engine = index.as_query_engine()
    #Option 1: Enable response_mode="refine" with strict checking
    #query_engine = index.as_query_engine(response_mode="refine")
    
    #Option 2: Use SimilarityPostprocessor to filter out low-relevance results
    '''from llama_index.core.postprocessor import SimilarityPostprocessor

query_engine = index.as_query_engine(
    similarity_top_k=3,
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.7)  # Only include results above 0.7 similarity
    ]
)
'''
    test_response = query_engine.query("What is this documentation about?")
    print(f"✅ Test successful!")
    print(f"📝 Response preview: {str(test_response)[:200]}...")
    print("\n🎉 Your FREE local RAG chatbot is ready to use!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    print("\n🔧 TROUBLESHOOTING STEPS:")
    
    if "memory" in str(e).lower():
        print("💡 Memory issue - try these smaller models:")
        print("   ollama pull llama3.2:1b")
        print("   ollama pull phi3:mini")
        print("   ollama pull qwen2:1.5b")
    elif "connection" in str(e).lower() or "not found" in str(e).lower():
        print("💡 Connection/Model issue:")
        print("   1. Check: ollama list")
        print("   2. Install: ollama pull llama3.2:1b")
        print("   3. Restart: ollama serve")
    else:
        print("💡 Other possible issues:")
        print("   - Close other applications to free RAM")
        print("   - Restart Ollama service")
        print("   - Check firewall settings")

print("\n🔧 SYSTEM INFO:")
print(f"   Project Drive: {os.getcwd()[0]}")
print(f"   Storage Path: {os.path.abspath('./storage')}")
print(f"   Model Cache: {os.path.abspath('./model_cache')}")
print(f"   Ollama Endpoint: http://localhost:11434")
print(f"   Selected Model: {model_to_use}")

print("\n📋 RECOMMENDED NEXT STEPS:")
print("   1. If this failed, run: ollama pull llama3.2:1b")
print("   2. Wait for download to complete")
print("   3. Run this script again")
print("   4. The 1B model only needs ~2GB RAM instead of 30GB")

#dev-docs-chatbot\Scripts\activate
#ollama pull llama3.2   or #ollama pull llama3.2:3b-instruct-q4_0
#python app2.py
#uvicorn main:app --reload