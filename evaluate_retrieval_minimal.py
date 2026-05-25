import os
import certifi
import torch

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

MONGO_URI = os.getenv("MONGO_URI")

DB_NAME = "legal_rag"
COLLECTION_NAME = "chunks"

TOP_K = 5

MODEL_NAME = "BAAI/bge-small-en-v1.5"


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"⚡ Using device: {device}")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("🧠 Loading embedding model...")

embedder = SentenceTransformer(
    MODEL_NAME,
    device=device
)


# ============================================================
# CONNECT TO MONGODB
# ============================================================

print("📥 Connecting to MongoDB Atlas...")

client = MongoClient(
    MONGO_URI,
    tlsCAFile=certifi.where()
)

db = client[DB_NAME]

collection = db[COLLECTION_NAME]


# ============================================================
# TEST QUERIES
# ============================================================

queries = [
    "governing law clause",
    "termination rights",
    "renewal term",
    "confidentiality obligations",
    "effective date"
]


# ============================================================
# SEARCH FUNCTION
# ============================================================

def search(query, top_k=5):

    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True
    )[0].tolist()

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": top_k
            }
        },
        {
            "$project": {
                "text": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }
    ]

    results = list(
        collection.aggregate(pipeline)
    )

    return results


# ============================================================
# RUN EVALUATION
# ============================================================

print("\n🚀 Running semantic retrieval test...\n")

for query in queries:

    print("================================================")
    print(f"🔎 QUERY: {query}")
    print("================================================")

    results = search(query, TOP_K)

    if not results:
        print("❌ No results found\n")
        continue

    for idx, result in enumerate(results):

        score = result["score"]

        text = result["text"][:300]

        print(f"\nResult #{idx + 1}")
        print(f"Score: {score:.4f}")

        print(f"Preview:\n{text}...\n")