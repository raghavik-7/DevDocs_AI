#!/usr/bin/env python3
"""
Storage Debug Script - Analyze and fix storage issues
"""

import os
import json
import chromadb
from llama_index.core import StorageContext, load_index_from_storage

def analyze_storage():
    """Analyze what's in the storage directory"""
    print("🔍 STORAGE ANALYSIS")
    print("=" * 50)
    
    if not os.path.exists('./storage'):
        print("❌ No storage directory found!")
        return
    
    contents = os.listdir('./storage')
    print(f"📁 Storage contents: {contents}")
    
    # Check for LlamaIndex default storage files
    default_files = ['docstore.json', 'index_store.json', 'graph_store.json', 'image__vector_store.json']
    found_default = [f for f in default_files if f in contents]
    
    if found_default:
        print(f"✅ Found LlamaIndex default storage files: {found_default}")
        
        # Try to load from default storage
        try:
            print("🔄 Testing default storage load...")
            storage_context = StorageContext.from_defaults(persist_dir='./storage')
            index = load_index_from_storage(storage_context)
            print("✅ Default storage loads successfully!")
            
            # Test a query
            query_engine = index.as_query_engine()
            response = query_engine.query("What is this about?")
            print(f"✅ Query test successful: {str(response)[:100]}...")
            
        except Exception as e:
            print(f"❌ Default storage load failed: {e}")
    
    # Check for ChromaDB files
    if 'chroma.sqlite3' in contents:
        print("✅ Found ChromaDB files")
        
        try:
            print("🔄 Testing ChromaDB load...")
            chroma_client = chromadb.PersistentClient(path="./storage")
            collections = chroma_client.list_collections()
            print(f"📋 ChromaDB collections: {[c.name for c in collections]}")
            
            if collections:
                # Try to get the first collection
                collection = collections[0]
                print(f"📊 Collection '{collection.name}' has {collection.count()} items")
                
        except Exception as e:
            print(f"❌ ChromaDB analysis failed: {e}")
    
    print("\n" + "=" * 50)

def recommend_solution():
    """Recommend what to do based on the analysis"""
    print("💡 RECOMMENDATIONS")
    print("=" * 50)
    
    if not os.path.exists('./storage'):
        print("1. Run app2.py to create the index first")
        return
    
    contents = os.listdir('./storage')
    
    # Check what storage types exist
    has_default = any(f.endswith('.json') for f in contents)
    has_chroma = 'chroma.sqlite3' in contents
    
    if has_default and has_chroma:
        print("⚠️  You have BOTH storage types! This causes conflicts.")
        print("   Recommended: Use the default storage (it's simpler)")
        print("   Action: Use the updated main.py - it tries default storage first")
        
    elif has_default:
        print("✅ You have default LlamaIndex storage")
        print("   Action: Use default storage loading in main.py")
        
    elif has_chroma:
        print("✅ You have ChromaDB storage")
        print("   Action: Use ChromaDB loading in main.py")
        
    else:
        print("❌ No valid storage found")
        print("   Action: Run app2.py to create the index")

def clean_storage():
    """Clean up conflicting storage files"""
    print("🧹 STORAGE CLEANUP")
    print("=" * 50)
    
    if not os.path.exists('./storage'):
        print("No storage to clean")
        return
    
    contents = os.listdir('./storage')
    has_default = any(f.endswith('.json') for f in contents)
    has_chroma = 'chroma.sqlite3' in contents
    
    if has_default and has_chroma:
        choice = input("You have both storage types. Keep which one? (default/chroma): ").lower()
        
        if choice == 'chroma':
            # Remove default files
            for f in contents:
                if f.endswith('.json'):
                    os.remove(f'./storage/{f}')
                    print(f"🗑️  Removed {f}")
        elif choice == 'default':
            # Remove ChromaDB files
            for f in contents:
                if 'chroma' in f.lower() or f.startswith('4a634675'):
                    path = f'./storage/{f}'
                    if os.path.isdir(path):
                        import shutil
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                    print(f"🗑️  Removed {f}")
        
        print("✅ Storage cleaned up!")

if __name__ == "__main__":
    print("🔧 LlamaIndex Storage Debugger")
    print("="*50)
    
    analyze_storage()
    recommend_solution()
    
    cleanup_choice = input("\nDo you want to clean up conflicting storage? (y/n): ")
    if cleanup_choice.lower() == 'y':
        clean_storage()
    
    print("\n🚀 Try running your FastAPI server now!")