import os
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
    
    # Check total documents to make sure the database is not empty
    total_docs = collection.count_documents({})
    print(f"   Database contains {total_docs} total chunks.")
    if total_docs == 0:
        print("❌ Error: MongoDB collection is empty. Run indexing first.")
        return

    # 3. Loading Dataset QAs
    print("\n📥 Loading CUAD dataset...")
    dataset = load_cuad_dataset()

    # Filter out valid QA pairs (where answer text is non-empty)
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
    
    # Take a representative subset for the diagnostic run
    test_suite = qa_pairs[:TEST_LIMIT]
    print(f"🚀 Running diagnostics on {len(test_suite)} samples...")

    ranks = []  # Store the rank (1-10) where the answer was found, or -1 if not found
    failures = [] # Diagnostic records for failed queries
    existence_checks = 0 # Check if the answer text actually exists in the database chunks at all
    
    for i, qa in enumerate(test_suite):
        query_text = qa["question"]
        expected_answer = qa["answer"]
        expected_clean = expected_answer.strip().lower()

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
            print(f"   ⚠️ Query {i+1} failed: {e}")
            continue

        ranks.append(found_rank)

        # 3. Check if the answer text actually exists in *any* database document (Index Coverage Check)
        in_index = False
        if found_rank == -1:
            # Query the database directly to check if the answer is contained in any document
            exists = collection.find_one({"text": {"$regex": expected_answer, "$options": "i"}})
            if exists:
                in_index = True
                existence_checks += 1
            
            # Store failure for diagnostic reporting
            failures.append({
                "query": query_text,
                "expected": expected_answer,
                "in_index": in_index,
                "retrieved_top": retrieved_texts[0] if retrieved_texts else "[No document retrieved]",
                "top_score": scores[0] if scores else 0.0
            })
        else:
            existence_checks += 1

    # =====================================================================
    # 📊 DIAGNOSTIC REPORT GENERATION
    # =====================================================================
    total_runs = len(ranks) if ranks else 1
    recall_1 = sum(1 for r in ranks if r == 1) / total_runs * 100
    recall_3 = sum(1 for r in ranks if 1 <= r <= 3) / total_runs * 100
    recall_5 = sum(1 for r in ranks if 1 <= r <= 5) / total_runs * 100
    recall_10 = sum(1 for r in ranks if 1 <= r <= 10) / total_runs * 100
    index_coverage = existence_checks / total_runs * 100

    print("\n" + "="*80)
    print("📊 RECALL & RETRIEVAL DIAGNOSTIC REPORT")
    print("="*80)
    print(f"🔹 Evaluated Queries      : {total_runs}")
    print(f"🔹 Index Coverage         : {index_coverage:.2f}% (Answer text exists in DB chunks)")
    print("-"*80)
    print(f"🎯 Recall@1 (Accuracy)    : {recall_1:.2f}%  <-- (True answer is at Rank 1)")
    print(f"🎯 Recall@3               : {recall_3:.2f}%  <-- (True answer is in Ranks 1-3)")
    print(f"🎯 Recall@5               : {recall_5:.2f}%  <-- (True answer is in Ranks 1-5)")
    print(f"🎯 Recall@10              : {recall_10:.2f}%  <-- (True answer is in Ranks 1-10)")
    print("="*80)

    # 4. Deep Diagnostic Failure Analysis
    print("\n🔍 DEEP DIAGNOSTIC ANALYSIS OF FAILURES:")
    print("="*80)
    
    if not failures:
        print("🎉 Awesome! Zero retrieval failures in this test run.")
        return

    # Print first 3 distinct failure cases
    for idx, f in enumerate(failures[:3], 1):
        print(f"\n❌ Failure Case #{idx}:")
        print(f"   ❓ Query: \"{f['query']}\"")
        print(f"   🎯 Expected Answer: \"{f['expected']}\"")
        print(f"   📂 Indexed? {'Yes (but vector search missed it)' if f['in_index'] else 'No (this text is missing from your indexed database!)'}")
        print(f"   🏷️ Rank 1 Score: {f['top_score']:.4f}")
        print(f"   📄 Top Chunk Retrieved:")
        print(f"      \"{f['retrieved_top'][:300]}...\"")
        
        # Diagnostic Recommendation
        print("   💡 DIAGNOSIS:")
        if not f['in_index']:
            print("      -> The answer text was NOT indexed. Reason: The contract containing this answer was either not parsed or fell outside your index slice.")
        else:
            print("      -> The correct chunk exists in the DB, but Vector Search preferred a different chunk.")
            print("         Reason: Legal boilerplate clauses (liability limits, notices) look extremely similar semantically, confuse the embedding model, and rank higher.")
        print("-" * 80)

if __name__ == "__main__":
    main()
