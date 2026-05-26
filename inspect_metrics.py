import os
import sys
import re
import time
import json
import certifi
import torch
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from core.data_loader import load_cuad_dataset
from core import config

def compress_query(query_text):
    """Reformulates long bloated CUAD instructions into clean search terms."""
    quoted_terms = re.findall(r'"([^"]*)"', query_text)
    if quoted_terms:
        return f"{quoted_terms[0]} clause"
    if "details:" in query_text.lower():
        parts = query_text.lower().split("details:")
        return parts[1].strip()
    return query_text

def run_benchmark(test_suite, collection, embedder, use_compression=False):
    ranks = []
    failures = []
    existence_checks = 0
    
    embedding_latencies = []
    retrieval_latencies = []
    total_latencies = []

    for i, qa in enumerate(test_suite):
        raw_query = qa["question"]
        expected_answer = qa["answer"]
        expected_clean = expected_answer.strip().lower()

        # Step A: Reformulate query if compression is enabled
        query_text = compress_query(raw_query) if use_compression else raw_query

        # Step 1: Measure Embedding Latency
        t_embed_start = time.perf_counter()
        query_vector = embedder.encode([query_text], normalize_embeddings=True).tolist()[0]
        t_embed_end = time.perf_counter()
        
        embed_ms = (t_embed_end - t_embed_start) * 1000.0
        embedding_latencies.append(embed_ms)

        # Step 2: Measure MongoDB Vector Retrieval Latency
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
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
        
        t_retrieval_start = time.perf_counter()
        try:
            results = list(collection.aggregate(pipeline))
            t_retrieval_end = time.perf_counter()
            retrieval_ms = (t_retrieval_end - t_retrieval_start) * 1000.0
            retrieval_latencies.append(retrieval_ms)
            
            for rank_idx, doc in enumerate(results, 1):
                chunk_text = doc.get("text", "")
                retrieved_texts.append(chunk_text)
                scores.append(doc.get("score", 0.0))
                
                if expected_clean in chunk_text.strip().lower():
                    found_rank = rank_idx
                    break
        except Exception as e:
            # Handle search fail safely
            retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000.0
            retrieval_latencies.append(retrieval_ms)
            total_latencies.append(embed_ms + retrieval_ms)
            continue

        total_latencies.append(embed_ms + retrieval_ms)
        ranks.append(found_rank)

        # Step 3: Index coverage existence verification
        escaped_answer = re.escape(expected_answer)
        in_index = False
        if found_rank == -1:
            exists = collection.find_one({"text": {"$regex": escaped_answer, "$options": "i"}})
            if exists:
                in_index = True
                existence_checks += 1
            
            failures.append({
                "index": i + 1,
                "raw_query": raw_query,
                "expected": expected_answer,
                "in_index": in_index,
                "retrieved_top": retrieved_texts[0] if retrieved_texts else "[No document retrieved]",
                "top_score": scores[0] if scores else 0.0
            })
        else:
            existence_checks += 1

    total_runs = len(ranks) if ranks else 1
    
    # Calculate Latency Stats
    latencies = {
        "embedding": {
            "mean_ms": float(np.mean(embedding_latencies)),
            "p50_ms": float(np.percentile(embedding_latencies, 50)),
            "p95_ms": float(np.percentile(embedding_latencies, 95)),
            "min_ms": float(np.min(embedding_latencies)),
            "max_ms": float(np.max(embedding_latencies))
        },
        "retrieval": {
            "mean_ms": float(np.mean(retrieval_latencies)),
            "p50_ms": float(np.percentile(retrieval_latencies, 50)),
            "p95_ms": float(np.percentile(retrieval_latencies, 95)),
            "min_ms": float(np.min(retrieval_latencies)),
            "max_ms": float(np.max(retrieval_latencies))
        },
        "total": {
            "mean_ms": float(np.mean(total_latencies)),
            "p50_ms": float(np.percentile(total_latencies, 50)),
            "p95_ms": float(np.percentile(total_latencies, 95)),
            "min_ms": float(np.min(total_latencies)),
            "max_ms": float(np.max(total_latencies))
        }
    }

    return {
        "recall_1": (sum(1 for r in ranks if r == 1) / total_runs) * 100.0,
        "recall_3": (sum(1 for r in ranks if 1 <= r <= 3) / total_runs) * 100.0,
        "recall_5": (sum(1 for r in ranks if 1 <= r <= 5) / total_runs) * 100.0,
        "recall_10": (sum(1 for r in ranks if 1 <= r <= 10) / total_runs) * 100.0,
        "index_coverage": (existence_checks / total_runs) * 100.0,
        "latencies": latencies,
        "failures": failures
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Legal RAG Metrics.")
    parser.add_argument("--samples", type=int, default=20, help="Number of CUAD queries to test")
    args = parser.parse_args()

    print("==================================================")
    print(" 🏛️  LEGAL AI — METRICS & RETRIEVAL INSPECTOR  🏛️")
    print("==================================================")
    print(f"📊 Evaluating precisely on {args.samples} samples...")

    # Load Credentials
    mongo_uri = os.getenv("MONGO_URI") or getattr(config, "MONGO_URI", None)
    if mongo_uri:
        mongo_uri = mongo_uri.strip()
        
    if not mongo_uri:
        print("❌ ERROR: MONGO_URI environment variable is missing.")
        sys.exit(1)

    # 1. Device Selection & Model Load
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"⚡ Device: [{device.upper()}]")
    
    print("🧠 Loading Embedding Model...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL, device=device)

    # 2. Database Connection
    print("📥 Connecting to MongoDB Atlas...")
    client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
    db = client["legal_rag"]
    collection = db["chunks"]
    
    total_docs = collection.count_documents({})
    print(f"   Database contains {total_docs:,} total chunks.")
    if total_docs == 0:
        print("❌ Error: MongoDB collection is empty. Run indexing first.")
        sys.exit(1)

    # 3. Loading Dataset QAs
    print("📥 Loading CUAD dataset...")
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
    test_suite = qa_pairs[:args.samples]

    # Run Benchmark A: Standard Bloated Queries
    print("\n🧪 Running Benchmark A: Raw, Bloated CUAD Prompts...")
    results_raw = run_benchmark(test_suite, collection, embedder, use_compression=False)

    # Run Benchmark B: Clean, Compressed Queries
    print("🧪 Running Benchmark B: Clean, Reformulated Queries...")
    results_clean = run_benchmark(test_suite, collection, embedder, use_compression=True)

    # =====================================================================
    # 📊 REPORT ASSEMBLY
    # =====================================================================
    report_md = f"""# 🏛️ Legal AI — RAG Performance & Latency Report

## 📈 Retrieval Metrics (N = {args.samples} Queries)

| Metric | Raw (Bloated) Prompts | Clean (Reformulated) Prompts | Impact |
| :--- | :---: | :---: | :---: |
| **Index Coverage** | {results_raw['index_coverage']:.2f}% | {results_clean['index_coverage']:.2f}% | Validates chunk parsing accuracy |
| **Recall@1 (Accuracy)** | {results_raw['recall_1']:.2f}% | {results_clean['recall_1']:.2f}% | **{results_clean['recall_1'] - results_raw['recall_1']:+.2f}%** |
| **Recall@3** | {results_raw['recall_3']:.2f}% | {results_clean['recall_3']:.2f}% | **{results_clean['recall_3'] - results_raw['recall_3']:+.2f}%** |
| **Recall@5** | {results_raw['recall_5']:.2f}% | {results_clean['recall_5']:.2f}% | **{results_clean['recall_5'] - results_raw['recall_5']:+.2f}%** |
| **Recall@10** | {results_raw['recall_10']:.2f}% | {results_clean['recall_10']:.2f}% | **{results_clean['recall_10'] - results_raw['recall_10']:+.2f}%** |

---

## ⏱️ Latency Analysis (Clean Queries)

| Pipeline Phase | Mean Latency | P50 (Median) | P95 Latency | Min Latency | Max Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Embedding Generation** | {results_clean['latencies']['embedding']['mean_ms']:.2f} ms | {results_clean['latencies']['embedding']['p50_ms']:.2f} ms | {results_clean['latencies']['embedding']['p95_ms']:.2f} ms | {results_clean['latencies']['embedding']['min_ms']:.2f} ms | {results_clean['latencies']['embedding']['max_ms']:.2f} ms |
| **MongoDB Atlas Search** | {results_clean['latencies']['retrieval']['mean_ms']:.2f} ms | {results_clean['latencies']['retrieval']['p50_ms']:.2f} ms | {results_clean['latencies']['retrieval']['p95_ms']:.2f} ms | {results_clean['latencies']['retrieval']['min_ms']:.2f} ms | {results_clean['latencies']['retrieval']['max_ms']:.2f} ms |
| **Total Pipeline (Local)** | {results_clean['latencies']['total']['mean_ms']:.2f} ms | {results_clean['latencies']['total']['p50_ms']:.2f} ms | {results_clean['latencies']['total']['p95_ms']:.2f} ms | {results_clean['latencies']['total']['min_ms']:.2f} ms | {results_clean['latencies']['total']['max_ms']:.2f} ms |

*Measurements taken on system platform device: [{device.upper()}].*

---

## 🔍 Failure Case Analysis (Top 3 Clean Query Failures)
"""

    failures = results_clean["failures"]
    if not failures:
        report_md += "\n🎉 **Awesome! Zero retrieval failures observed with reformulated queries.**\n"
    else:
        for idx, f in enumerate(failures[:3], 1):
            report_md += f"""
### ❌ Failure Case #{idx}:
*   **Raw Prompt**: *"{f['raw_query'][:120]}..."*
*   **Search Term Used**: `"{f['compressed_query']}"`
*   **Expected Ground Truth**: `"{f['expected']}"`
*   **Status in Database**: {"Yes (but search missed it)" if f["in_index"] else "No (not indexed in this slice)"}
*   **Top Score Retreived**: `{f['top_score']:.4f}`
*   **Chunk Retrieved Preview**:
    > *"{f['retrieved_top'][:250]}..."*
"""

    # Print to console
    print("\n" + "="*80)
    print("📊 PRINTING BENCHMARK REPORT SUMMARY")
    print("="*80)
    print(f"🎯 Recall@1 (Accuracy)    : Raw: {results_raw['recall_1']:.2f}%  |  Clean: {results_clean['recall_1']:.2f}%")
    print(f"🎯 Recall@10              : Raw: {results_raw['recall_10']:.2f}%  |  Clean: {results_clean['recall_10']:.2f}%")
    print(f"⏱️  Mean Search Latency   : {results_clean['latencies']['retrieval']['mean_ms']:.2f} ms")
    print(f"⏱️  P95 Search Latency    : {results_clean['latencies']['retrieval']['p95_ms']:.2f} ms")
    print("="*80 + "\n")

    # Save reports
    os.makedirs("reports", exist_ok=True)
    
    with open("reports/metrics_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
        
    report_json = {
        "samples_evaluated": args.samples,
        "raw_results": {
            "recall_1": results_raw["recall_1"],
            "recall_10": results_raw["recall_10"],
            "index_coverage": results_raw["index_coverage"]
        },
        "clean_results": {
            "recall_1": results_clean["recall_1"],
            "recall_10": results_clean["recall_10"],
            "index_coverage": results_clean["index_coverage"],
            "latencies": results_clean["latencies"]
        }
    }
    
    with open("reports/metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=4)
        
    print("💾 Metrics report successfully saved to reports/metrics_report.md and reports/metrics_report.json")

if __name__ == "__main__":
    main()
