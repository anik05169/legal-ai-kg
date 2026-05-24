import os
import time
import certifi
import torch
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from core.data_loader import load_cuad_dataset

# =====================================================================
# ⚙️ CONFIGURATION
# =====================================================================
# Load .env file manually if it exists to avoid python-dotenv dependency
if os.path.exists(".env"):
    try:
        with open(".env", "r") as f:
            for line in f:
                if line.strip() and not line.strip().startswith("#") and "=" in line:
                    key, val = line.strip().split("=", 1)
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")
    except Exception as e:
        print(f"⚠️ Warning: Failed to load .env file manually: {e}")

MONGO_URI = os.getenv("MONGO_URI") or "mongodb+srv://MohmedAnik:<password>@cluster0.siogtjt.mongodb.net/?appName=Cluster0"
DB_NAME = "legal_rag"
COLLECTION_NAME = "chunks"
INDEX_NAME = "vector_index"


BENCHMARK_CONTRACTS = 3  # Number of contracts to index for the speed benchmark
CHUNK_SIZE = 150
OVERLAP = 30

def chunk_text(text, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []
    if len(words) == 0:
        return chunks
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def main():
    print("=====================================================================")
    print("📊 LEGAL AI GRAPHRAG BENCHMARKING ENGINE")
    print("=====================================================================")

    # 1. Setup & Connection
    print("⚡ Device selection...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"   Using device: [{device.upper()}]")

    print("\n🧠 Loading Embedding Model...")
    model_load_start = time.perf_counter()
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    model_load_time = time.perf_counter() - model_load_start
    print(f"   Loaded model in: {model_load_time:.2f} seconds")

    print("\n📥 Connecting to MongoDB Atlas...")
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    print("   Connected successfully.")

    # 2. Loading Dataset
    print("\n📥 Loading CUAD dataset...")
    dataset_load_start = time.perf_counter()
    dataset = load_cuad_dataset()
    dataset_load_time = time.perf_counter() - dataset_load_start
    print(f"   Dataset loaded in: {dataset_load_time:.2f} seconds")

    # Group questions by contract text to set up queries
    contracts_dict = {}
    for item in dataset:
        context = item['context']
        question = item['question']
        
        # Extract the ground truth answer text
        answers = item.get('answers', {})
        answer_texts = answers.get('text', []) if isinstance(answers, dict) else []
        answer = answer_texts[0] if len(answer_texts) > 0 else ""
        
        if context not in contracts_dict:
            contracts_dict[context] = []
        if answer:  # Only track questions that have a valid answer
            contracts_dict[context].append({
                "question": question,
                "answer": answer
            })

    unique_contracts = list(contracts_dict.keys())
    selected_contracts = unique_contracts[:BENCHMARK_CONTRACTS]

    # =====================================================================
    # ⏱️ PHASE 1: INDEXING LATENCY BENCHMARK
    # =====================================================================
    print("\n🚀 Starting Indexing Speed Benchmark (using 3 contracts)...")
    collection.delete_many({}) # Clear before benchmark
    
    total_chunk_time = 0.0
    total_embed_time = 0.0
    total_db_write_time = 0.0
    total_chunks = 0

    for idx, contract_text in enumerate(selected_contracts):
        # 1. Text Chunking
        chunk_start = time.perf_counter()
        chunks = chunk_text(contract_text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        chunk_time = time.perf_counter() - chunk_start
        total_chunk_time += chunk_time
        total_chunks += len(chunks)

        # 2. Embedding Generation
        embed_start = time.perf_counter()
        embeddings = embedder.encode(chunks).tolist()
        embed_time = time.perf_counter() - embed_start
        total_embed_time += embed_time

        # 3. Database Write
        payload = []
        for c_idx, text_passage in enumerate(chunks):
            payload.append({
                "text": text_passage,
                "embedding": embeddings[c_idx],
                "metadata": {"contract_index": idx, "chunk_index": c_idx}
            })
        
        db_start = time.perf_counter()
        if payload:
            collection.insert_many(payload)
        db_write_time = time.perf_counter() - db_start
        total_db_write_time += db_write_time

    avg_chunk_latency = (total_chunk_time / BENCHMARK_CONTRACTS) * 1000
    avg_embed_latency = (total_embed_time / total_chunks) * 1000
    avg_db_latency = (total_db_write_time / total_chunks) * 1000
    total_pipeline_time = total_chunk_time + total_embed_time + total_db_write_time

    print(f"\n📊 --- INDEXING SPEED RESULTS ---")
    print(f"   Total Chunks Created & Indexed: {total_chunks}")
    print(f"   Avg Chunking Time per Contract: {avg_chunk_latency:.2f} ms")
    print(f"   Avg Embedding Generation per Chunk: {avg_embed_latency:.2f} ms")
    print(f"   Avg MongoDB Atlas Upload per Chunk: {avg_db_latency:.2f} ms")
    print(f"   Total Indexing Speed: {total_chunks / total_pipeline_time:.2f} chunks/second")

    # =====================================================================
    # ⏱️ PHASE 2: RETRIEVAL SPEED & ACCURACY BENCHMARK
    # =====================================================================
    print("\n⏳ Waiting 5 seconds to ensure MongoDB Atlas Vector index updates...")
    time.sleep(5)

    print("\n🚀 Starting Retrieval Speed & Accuracy Benchmark...")
    
    # Gather test queries that belong to the contracts we just indexed
    test_queries = []
    for idx, contract_text in enumerate(selected_contracts):
        qas = contracts_dict.get(contract_text, [])
        # Sample up to 10 queries per contract to run a broad test
        test_queries.extend(qas[:10])

    if not test_queries:
        print("⚠️ No queries with answers found in these contracts. Accuracy cannot be tested.")
        return

    print(f"   Collected {len(test_queries)} test queries with ground-truth answers.")
    
    total_query_time = 0.0
    total_retrieval_time = 0.0
    scores = []
    hits = 0

    for query_item in test_queries:
        query_text = query_item["question"]
        expected_answer = query_item["answer"]

        # 1. Embedding Query
        query_embed_start = time.perf_counter()
        query_vector = embedder.encode([query_text]).tolist()[0]
        query_embed_time = time.perf_counter() - query_embed_start
        total_query_time += query_embed_time

        # 2. Vector Search Retrieval
        pipeline = [
            {
                "$vectorSearch": {
                    "index": INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 15,
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

        retrieval_start = time.perf_counter()
        try:
            results = list(collection.aggregate(pipeline))
            retrieval_time = time.perf_counter() - retrieval_start
            total_retrieval_time += retrieval_time

            if results:
                retrieved_chunk = results[0].get("text", "")
                score = results[0].get("score", 0.0)
                scores.append(score)

                # Check Accuracy: If the correct answer span is contained in the retrieved chunk
                if expected_answer.strip().lower() in retrieved_chunk.strip().lower():
                    hits += 1
        except Exception as e:
            # Handle the case where the Atlas Index is still building and not ready yet
            print(f"   ⚠️ Search query failed: {e}. (Atlas Index might still be building).")

    total_benchmarks = len(scores) if scores else 1
    avg_query_embed_latency = (total_query_time / len(test_queries)) * 1000
    avg_db_retrieval_latency = (total_retrieval_time / len(test_queries)) * 1000
    avg_score = sum(scores) / total_benchmarks if scores else 0
    accuracy = (hits / len(test_queries)) * 1000 / 10 if test_queries else 0 # In percentage (0-100)

    print(f"\n📊 --- RETRIEVAL SPEED & ACCURACY RESULTS ---")
    print(f"   Average Query Embedding Latency: {avg_query_embed_latency:.2f} ms")
    print(f"   Average MongoDB Cloud Retrieval Latency: {avg_db_retrieval_latency:.2f} ms")
    print(f"   Average Vector Search Cosine Similarity Score: {avg_score:.4f}")
    print(f"   Average Retrieval Accuracy (Recall@1): {hits}/{len(test_queries)} ({accuracy:.2f}%)")
    print("=====================================================================")

if __name__ == "__main__":
    main()
