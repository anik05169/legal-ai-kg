import os
import sys
import uuid
import hashlib
import certifi
import torch

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# Ensure project root is in path when run as a standalone script
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.core.data_loader import load_cuad_dataset


# =====================================================================
# ⚙️ CONFIGURATION
# =====================================================================

MONGO_URI = os.getenv("MONGO_URI")

DB_NAME = "legal_rag"
COLLECTION_NAME = "chunks"

# Better settings for legal contracts
CHUNK_SIZE = 500
OVERLAP = 100

# Number of unique CUAD contracts to index
NUM_CONTRACTS = 1000

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Batch size for embeddings
BATCH_SIZE = 32


# =====================================================================
# 🛠️ HELPER FUNCTIONS
# =====================================================================

def chunk_text(text, chunk_size=500, overlap=100):
    """
    Splits legal contract text into overlapping chunks.
    """

    words = text.split()

    if not words:
        return []

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def generate_contract_id(contract_text):
    """
    Generates stable deterministic contract ID.
    """

    return hashlib.md5(contract_text.encode("utf-8")).hexdigest()


# =====================================================================
# 🚀 MAIN PIPELINE
# =====================================================================

def main():

    # -------------------------------------------------------------
    # Validate Mongo URI
    # -------------------------------------------------------------

    if not MONGO_URI:
        raise ValueError(
            "❌ MONGO_URI environment variable missing."
        )

    # -------------------------------------------------------------
    # Device Selection
    # -------------------------------------------------------------

    print("⚡ Selecting device...")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"✅ Using device: [{device.upper()}]")

    # -------------------------------------------------------------
    # Load Embedding Model
    # -------------------------------------------------------------

    print("\n🧠 Loading embedding model...")

    embedder = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device
    )

    # -------------------------------------------------------------
    # MongoDB Connection
    # -------------------------------------------------------------

    print("\n📥 Connecting to MongoDB Atlas...")

    client = MongoClient(
        MONGO_URI,
        tlsCAFile=certifi.where()
    )

    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # -------------------------------------------------------------
    # Clear Existing Collection
    # -------------------------------------------------------------

    print("🧹 Clearing old indexed chunks...")

    collection.delete_many({})

    # -------------------------------------------------------------
    # Load Dataset
    # -------------------------------------------------------------

    print("\n📥 Loading CUAD dataset...")

    dataset = load_cuad_dataset()

    # -------------------------------------------------------------
    # Preserve Ordering (IMPORTANT)
    # -------------------------------------------------------------

    unique_contracts = list(
        dict.fromkeys(dataset["context"])
    )

    print(
        f"📊 Total unique contracts found: "
        f"{len(unique_contracts)}"
    )

    # -------------------------------------------------------------
    # Select Subset
    # -------------------------------------------------------------

    selected_contracts = unique_contracts[:NUM_CONTRACTS]

    print(
        f"🚀 Indexing {len(selected_contracts)} contracts..."
    )

    total_chunks_indexed = 0

    # -------------------------------------------------------------
    # Process Contracts
    # -------------------------------------------------------------

    for doc_idx, contract_text in enumerate(selected_contracts):

        print(
            f"\n📄 Processing Contract "
            f"{doc_idx + 1}/{len(selected_contracts)}"
        )

        contract_id = generate_contract_id(contract_text)

        # ---------------------------------------------------------
        # Chunk Contract
        # ---------------------------------------------------------

        chunks = chunk_text(
            contract_text,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP
        )

        print(
            f"   -> Generated {len(chunks)} chunks"
        )

        if not chunks:
            print("   ⚠️ Skipping empty contract")
            continue

        # ---------------------------------------------------------
        # Generate Embeddings
        # ---------------------------------------------------------

        print("   🧠 Generating embeddings...")

        embeddings = embedder.encode(
            chunks,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # ---------------------------------------------------------
        # Build Mongo Payload
        # ---------------------------------------------------------

        payload = []

        for chunk_idx, chunk_text_value in enumerate(chunks):

            payload.append({
                "_id": str(uuid.uuid4()),

                "text": chunk_text_value,

                "embedding": embeddings[chunk_idx].tolist(),

                "metadata": {
                    "contract_id": contract_id,
                    "contract_index": doc_idx,
                    "chunk_index": chunk_idx,
                    "word_count": len(
                        chunk_text_value.split()
                    )
                }
            })

        # ---------------------------------------------------------
        # Insert Into MongoDB
        # ---------------------------------------------------------

        if payload:

            collection.insert_many(payload)

            total_chunks_indexed += len(payload)

            print(
                f"   ✅ Indexed {len(payload)} chunks"
            )

    # =================================================================
    # FINAL STATS
    # =================================================================

    print("\n══════════════════════════════════════")
    print("🏆 INDEXING COMPLETE")
    print("══════════════════════════════════════")

    print(f"📦 Contracts indexed : {len(selected_contracts)}")
    print(f"📄 Total chunks      : {total_chunks_indexed}")

    # =================================================================
    # VECTOR SEARCH VERIFICATION
    # =================================================================

    print("\n🔍 Running verification vector search...")

    test_query = "governing law clause"

    query_vector = embedder.encode(
        [test_query],
        normalize_embeddings=True
    )[0].tolist()

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": 3
            }
        },
        {
            "$project": {
                "text": 1,
                "metadata": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }
    ]

    try:

        results = list(
            collection.aggregate(pipeline)
        )

        if not results:

            print(
                "⚠️ No vector search results returned."
            )

        else:

            print("\n🎉 Verification successful!")

            for idx, result in enumerate(results):

                print("\n----------------------------------")
                print(f"Result #{idx + 1}")
                print("----------------------------------")

                print(
                    f"Score : "
                    f"{result.get('score'):.4f}"
                )

                print(
                    f"Chunk : "
                    f"{result.get('metadata', {}).get('chunk_index')}"
                )

                preview = result.get("text", "")[:300]

                print(f"Preview:\n{preview}...")

    except Exception as e:

        print("\n❌ Vector search failed")
        print(str(e))


# =====================================================================
# ENTRYPOINT
# =====================================================================

if __name__ == "__main__":
    main()