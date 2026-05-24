import os
import re
import certifi
import torch
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from core.data_loader import load_cuad_dataset

# =====================================================================
# ⚙️ CONFIGURATION
# =====================================================================
# Load local .env manually if it exists to retrieve MONGO_URI
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

TEST_LIMIT = 30  # Analyze the first 30 queries for deep diagnostics

def compress_query(query_text):
    """
    Query Reformulation: Extracts the core legal concept from the bloated CUAD instruction.
    E.g. extracts 'Document Name' or the text after 'Details: ' to avoid prompt dilution.
    """
    # 1. Try to extract the clause name in quotes (e.g. "Document Name")
    quoted_terms = re.findall(r'"([^"]*)"', query_text)
    if quoted_terms:
        return f"{quoted_terms[0]} clause"
    
    # 2. Fall back to the Details section
    if "details:" in query_text.lower():
        parts = query_text.lower().split("details:")
        return parts[1].strip()
        
    return query_text

def run_recall_test(test_suite, collection, embedder, use_compression=False):
    ranks = []
    failures = []
    existence_checks = 0

    for i, qa in enumerate(test_suite):
        raw_query = qa["question"]
        expected_answer = qa["answer"]
        expected_clean = expected_answer.strip().lower()

        # Reformulate query if compression is enabled
        query_text = compress_query(raw_query) if use_compression else raw_query

        # 1. Generate query embedding
        query_vector = embedder.encode([query_text]).tolist()[0]

        # 2. Vector Search (fetching top-10 candidates)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 50,
                    "limit": 10
                }
            },
            {
                "$project": {
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        found_rank = -1
        retrieved_texts = []
        scores = []
        
        try:
            results = list(collection.aggregate(pipeline))
            for rank_idx, doc in enumerate(results, 1):
                chunk_text = doc.get("text", "")
                retrieved_texts.append(chunk_text)
                scores.append(doc.get("score", 0.0))
                
                # Check if correct answer is inside the chunk text
                if expected_clean in chunk_text.strip().lower():
                    found_rank = rank_idx
                    break
        except Exception as e:
            continue

        ranks.append(found_rank)

        # 3. Escape regex special characters to prevent index coverage match failures!
        escaped_answer = re.escape(expected_answer)
        in_index = False
        if found_rank == -1:
            exists = collection.find_one({"text": {"$regex": escaped_answer, "$options": "i"}})
            if exists:
                in_index = True
                existence_checks += 1
            
            failures.append({
                "raw_query": raw_query,
                "compressed_query": query_text,
                "expected": expected_answer,
                "in_index": in_index,
                "retrieved_top": retrieved_texts[0] if retrieved_texts else "[No document retrieved]",
                "top_score": scores[0] if scores else 0.0
            })
        else:
            existence_checks += 1

    total_runs = len(ranks) if ranks else 1
    recall_1 = sum(1 for r in ranks if r == 1) / total_runs * 100
    recall_3 = sum(1 for r in ranks if 1 <= r <= 3) / total_runs * 100
    recall_5 = sum(1 for r in ranks if 1 <= r <= 5) / total_runs * 100
    recall_10 = sum(1 for r in ranks if 1 <= r <= 10) / total_runs * 100
    index_coverage = existence_checks / total_runs * 100

    return {
        "recall_1": recall_1,
        "recall_3": recall_3,
        "recall_5": recall_5,
        "recall_10": recall_10,
        "index_coverage": index_coverage,
        "failures": failures
    }

def main():
    print("=====================================================================")
    print("🔍 LEGAL AI RAG RECALL DIAGNOSTIC UTILITY")
    print("=====================================================================")

    # 1. Device Selection & Model Load
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"⚡ Device: [{device.upper()}]")
    
    print("\n🧠 Loading Embedding Model...")
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    # 2. Database Connection
    print("\n📥 Connecting to MongoDB Atlas...")
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    total_docs = collection.count_documents({})
    print(f"   Database contains {total_docs} total chunks.")
    if total_docs == 0:
        print("❌ Error: MongoDB collection is empty. Run indexing first.")
        return

    # 3. Loading Dataset QAs
    print("\n📥 Loading CUAD dataset...")
    dataset = load_cuad_dataset()

    qa_pairs = []
    for item in dataset:
        context = item['context']
        question = item['question']
        answers = item.get('answers', {})
        answer_texts = answers.get('text', []) if isinstance(answers, dict) else []
        
        if answer_texts and answer_texts[0].strip():
            qa_pairs.append({
                "context": context,
                "question": question,
                "answer": answer_texts[0].strip()
            })

    print(f"   Found {len(qa_pairs)} queries with ground-truth answers.")
    test_suite = qa_pairs[:TEST_LIMIT]
    print(f"🚀 Running diagnostics on {len(test_suite)} samples...")

    # Run Benchmark A: Standard Bloated Queries
    print("\n🧪 Running Test A: Raw, Bloated CUAD Prompts...")
    results_raw = run_recall_test(test_suite, collection, embedder, use_compression=False)

    # Run Benchmark B: Clean, Compressed Queries
    print("🧪 Running Test B: Clean, Reformulated Queries...")
    results_clean = run_recall_test(test_suite, collection, embedder, use_compression=True)

    # =====================================================================
    # 📊 SIDE-BY-SIDE COMPARISON REPORT
    # =====================================================================
    print("\n" + "="*80)
    print("📊 RECALL & RETRIEVAL DIAGNOSTIC REPORT (SIDE-BY-SIDE)")
    print("="*80)
    print(f"🔹 Evaluated Queries      : {len(test_suite)}")
    print(f"🔹 True Index Coverage    : {results_clean['index_coverage']:.2f}% (Answer text exists in DB chunks)")
    print("-"*80)
    print(f"🎯 Recall@1 (Accuracy)    : Raw: {results_raw['recall_1']:.2f}%  |  Clean: {results_clean['recall_1']:.2f}%")
    print(f"🎯 Recall@3               : Raw: {results_raw['recall_3']:.2f}%  |  Clean: {results_clean['recall_3']:.2f}%")
    print(f"🎯 Recall@5               : Raw: {results_raw['recall_5']:.2f}%  |  Clean: {results_clean['recall_5']:.2f}%")
    print(f"🎯 Recall@10              : Raw: {results_raw['recall_10']:.2f}%  |  Clean: {results_clean['recall_10']:.2f}%")
    print("="*80)

    # 4. Deep Failure Analysis on Clean Queries
    failures = results_clean["failures"]
    print("\n🔍 DEEP DIAGNOSTIC ANALYSIS OF REMAINING FAILURES (Clean Queries):")
    print("="*80)
    
    if not failures:
        print("🎉 Awesome! Zero retrieval failures with reformulated queries.")
        return

    for idx, f in enumerate(failures[:3], 1):
        print(f"\n❌ Failure Case #{idx}:")
        print(f"   ❓ Raw Prompt: \"{f['raw_query'][:100]}...\"")
        print(f"   🎯 Compressed Search Term: \"{f['compressed_query']}\"")
        print(f"   🎯 Expected Answer: \"{f['expected']}\"")
        print(f"   📂 Indexed? {'Yes (but vector search missed it)' if f['in_index'] else 'No (this text fell outside your database slice)'}")
        print(f"   🏷️ Rank 1 Score: {f['top_score']:.4f}")
        print(f"   📄 Top Chunk Retrieved:")
        print(f"      \"{f['retrieved_top'][:220]}...\"")
        
        print("   💡 DIAGNOSIS:")
        if not f['in_index']:
            print("      -> The answer text was NOT indexed. Reason: The contract containing this answer was either not parsed or fell outside your index slice.")
        else:
            print("      -> The correct chunk exists in the DB, but Vector Search preferred a different chunk.")
            print("         Reason: Legal boilerplate clauses (liability limits, notices) look extremely similar semantically, confuse the embedding model, and rank higher.")
        print("-" * 80)

if __name__ == "__main__":
    main()

