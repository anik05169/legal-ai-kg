import os
import uuid
import torch
import certifi
from datasets import load_dataset
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# =====================================================================
# ⚙️ CONFIGURATION
# =====================================================================
# Pulls connection string securely from the environment variables set by GitHub secrets
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "legal_rag"
COLLECTION_NAME = "chunks"

CHUNK_SIZE = 150  # Words per chunk
OVERLAP = 30      # Overlapping words between chunks
NUM_CONTRACTS = 5 # Number of unique CUAD contracts to index

# =====================================================================
# 🛠️ HELPER FUNCTIONS
# =====================================================================
def chunk_text(text, chunk_size=150, overlap=30):
    """Splits a document text string into overlapping chunks of words."""
    words = text.split()
    chunks = []
    if len(words) == 0:
        return chunks
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# =====================================================================
# 🚀 MAIN PIPELINE
# =====================================================================
def main():
    if not MONGO_URI:
        raise ValueError("❌ MONGO_URI environment variable is missing. Set it in GitHub secrets or environment variables.")

    print("⚡ Device selection...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"   Using hardware accelerator: [{device.upper()}]")

    print("\n🧠 Loading embedding model...")
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    print("\n📥 Connecting to MongoDB Atlas cluster...")
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Optional: Clear existing collection for a clean slate
    print("🧹 Wiping clean old documents in cloud collection...")
    collection.delete_many({})

    print("\n📥 Loading CUAD dataset from Hugging Face...")
    dataset = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)
    
    # Extract unique legal contracts
    unique_contracts = list(set(dataset["context"]))
    print(f"📊 Total unique contracts found in CUAD dataset: {len(unique_contracts)}")
    
    selected_contracts = unique_contracts[:NUM_CONTRACTS]
    print(f"🚀 Selected {len(selected_contracts)} contracts for processing.")

    total_chunks_indexed = 0

    for doc_idx, contract_text in enumerate(selected_contracts):
        print(f"\n📄 Processing Contract {doc_idx + 1}/{len(selected_contracts)}...")
        
        # Chunk text
        chunks = chunk_text(contract_text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        print(f"   -> Created {len(chunks)} chunks from contract. Generating embeddings...")
        
        # Generate embeddings in batch
        embeddings = embedder.encode(chunks).tolist()
        
        # Build bulk insert payload
        payload = []
        for idx, text_passage in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            payload.append({
                "_id": chunk_id,
                "text": text_passage,
                "embedding": embeddings[idx],
                "metadata": {
                    "contract_index": doc_idx,
                    "chunk_index": idx,
                    "word_count": len(text_passage.split())
                }
            })
            
        # Bulk write to MongoDB
        if payload:
            collection.insert_many(payload)
            total_chunks_indexed += len(payload)
            print(f"   ✅ Successfully indexed {len(payload)} chunks in MongoDB Atlas.")

    print(f"\n🏆 Pipeline complete! Total chunks indexed in MongoDB: {total_chunks_indexed}")

    # =====================================================================
    # 🔍 VERIFY INDEX (MOCK QUERY RUN)
    # =====================================================================
    print("\n🔍 Running verification Vector Search query...")
    test_query = "What is the governing law of the agreement?"
    query_vector = embedder.encode([test_query]).tolist()[0]
    
    # Run MongoDB Vector Search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 10,
                "limit": 1
            }
        },
        {
            "$project": {
                "text": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    try:
        results = list(collection.aggregate(pipeline))
        if results:
            print("\n🎉 Verification Success! Best matching chunk found in Atlas:")
            print(f"   Match Score: {results[0].get('score'):.4f}")
            print(f"   Content Preview: {results[0].get('text')[:200]}...")
        else:
            print("\n⚠️ Vector Search completed, but returned 0 results. Ensure your Atlas Vector Search Index is fully build and 'Active'.")
    except Exception as e:
        print(f"\n❌ Vector Search query failed: {e}")

if __name__ == "__main__":
    main()
