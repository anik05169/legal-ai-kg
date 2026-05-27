import os
import sys
import time
import json
import argparse
import certifi
import torch
import pandas as pd
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    context_recall,
    faithfulness,
    AspectCritic
)

# --- FORCE IPV4 ONLY (Bypasses broken IPv6 routing) ---
import socket
orig_getaddrinfo = socket.getaddrinfo
def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = patched_getaddrinfo

# Add parent directory to path to ensure configs can be loaded if needed
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.core import config

def run_evaluation():
    parser = argparse.ArgumentParser(description="Evaluate Legal AI RAG Pipeline against a CSV benchmark.")
    parser.add_argument(
        "--csv", 
        type=str, 
        default="tests/benchmark.csv", 
        help="Path to the benchmark CSV file"
    )
    args = parser.parse_args()

    # --- 1. Load Credentials & Settings ---
    mongo_uri = os.getenv("MONGO_URI") or getattr(config, "MONGO_URI", None)
    if mongo_uri:
        mongo_uri = mongo_uri.strip()
        
    groq_api_key = os.getenv("GROQ_API_KEY") or getattr(config, "GROQ_API_KEY", None)
    if groq_api_key:
        groq_api_key = groq_api_key.strip()
        
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        openai_api_key = openai_api_key.strip()
    
    if not mongo_uri:
        print("❌ ERROR: MONGO_URI environment variable or configuration is missing.")
        sys.exit(1)
        
    if not groq_api_key:
        print("❌ ERROR: GROQ_API_KEY environment variable or configuration is missing.")
        sys.exit(1)
        
    if not os.path.exists(args.csv):
        print(f"❌ ERROR: Benchmark file not found at {args.csv}")
        sys.exit(1)

    print("==================================================")
    print("      LEGAL AI - RAG EVALUATION BENCHMARK")
    print("==================================================")
    print(f"Loading benchmark from: {args.csv}")
    
    # Read benchmark data
    try:
        test_df = pd.read_csv(args.csv, delimiter=";")
        print(f"Successfully loaded {len(test_df)} evaluation cases.")
    except Exception as e:
        print(f"❌ Failed to parse CSV: {e}")
        sys.exit(1)

    # --- 2. Hardware & Models Setup ---
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: [{device.upper()}]")

    print("Loading embedding model...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL, device=device)

    # --- 3. Connect to MongoDB Atlas ---
    print("Connecting to MongoDB Atlas...")
    try:
        mongo_client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
        db = mongo_client["legal_rag"]
        collection = db["chunks"]
        # Trigger quick operation to verify connection
        collection.find_one()
        print("✅ Successfully connected to MongoDB Atlas.")
    except Exception as e:
        print(f"❌ MongoDB Connection failed: {e}")
        sys.exit(1)

    # --- 4. Initialize Groq LLM Client ---
    print(f"Initializing Groq LLM Client ({config.GROQ_MODEL})...")
    llm_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key
    )

    # --- 5. Custom Aspect Critic Definition (From Reference) ---
    answer_satisfaction = AspectCritic(
        name="answer_satisfaction",
        definition="""You will evaluate an ANSWER to a legal QUESTION based on a provided SOLUTION.

Rate the answer on a scale from 0 to 1, where:
- 0 = incorrect, substantially incomplete, or misleading
- 1 = correct and sufficiently complete

Consider these evaluation criteria:
1. Factual correctness is paramount - the answer must not contradict the solution
2. The answer must address the core elements of the solution
3. Additional relevant information beyond the solution is acceptable and may enhance the answer
4. Technical legal terminology should be used appropriately if present in the solution
5. For quantitative legal analyses, accurate figures must be provided

### Examples:

QUESTION: What is the statute of limitations for breach of contract in California?
GROUND_TRUTH: In California, the statute of limitations for written contracts is 4 years, and for oral contracts is 2 years, as specified in California Code of Civil Procedure § 337 and § 339.
RESPONSE: California has a 4-year statute of limitations for written contract claims and 2 years for oral contracts.
RATING: 1

QUESTION: What is required for a valid contract?
GROUND_TRUTH: A valid contract requires offer, acceptance, consideration, legal capacity of the parties, legal purpose, and mutual assent (meeting of the minds).
RESPONSE: For a contract to be valid, there must be consideration exchanged between the parties.
RATING: 0

QUESTION: What contracts are governed by US law?
GROUND_TRUTH: There are 316 contracts governed by US law
RESPONSE: There are 316 contracts governed by US law. Here are a few examples:\n\n1. **Remarketing Agreement**:\n - **Parties**: Nuveen AMT-Free Municipal Credit Income Fund, Nuveen Fund Advisors, LLC, and a Remarketing Agent.\n - **Type**: Service\n - **Scope**: Remarketing of Series MuniFund Preferred Shares.\n - **Effective Date**: January 1, 2024\n\n2. **Joint Filing Agreement**:\n - **Parties**: Multiple parties including Athene Annuity and Life Company, Apollo Insurance Solutions Group...\t
RATING: 1

QUESTION: What is the total monetary value of all contracts with WebMD, Inc.?
GROUND_TRUTH: The total monetary value of all contracts with WebMD, Inc. is $5,000.\t
RESPONSE: There is only one contract involving WebMD, Inc., which is a Content License Agreement with Emdeon Corporation. This agreement allows Emdeon to use WebMD's anatomical line drawings in its software and the value is 5000
RATING: 1""",
    )

    questions = []
    ground_truths = []
    responses = []
    retrieved_contexts = []
    latencies = []

    # --- 6. Execute Vector RAG Pipeline ---
    print("\nStarting evaluation pipeline...")
    for idx, row in test_df.iterrows():
        question = row["question"]
        ground_truth = row["ground_truth"]
        
        print(f"[{idx+1}/{len(test_df)}] Query: '{question[:60]}...'")
        
        start_time = time.time()
        
        # A. Vector Retrieval
        query_embedding = embedder.encode([question], normalize_embeddings=True)[0].tolist()
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": 5
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
            context_list = [res["text"] for res in results]
        except Exception as e:
            print(f"  ❌ MongoDB Vector Search failed: {e}")
            context_list = []
            
        retrieved_contexts.append(context_list)
        
        # B. Groq Answer Generation
        final_context = "=== LEGAL TEXT EXCERPTS ===\n" + "\n---\n".join(context_list) if context_list else "No relevant context found."
        
        system_prompt = (
            "You are an expert legal AI assistant. Your job is to answer the user's question clearly "
            "and naturally using ONLY the provided context.\n"
            "RULES:\n"
            "1. Read the provided LEGAL TEXT EXCERPTS.\n"
            "2. Answer in a helpful, conversational tone.\n"
            "3. If the answer is not in the text, simply state that the provided document does not contain the answer.\n"
            "4. Do not provide external legal advice or hallucinate outside knowledge."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n{final_context}\n\nQUERY: {question}"}
        ]

        try:
            response = llm_client.chat.completions.create(
                model=config.GROQ_MODEL,
                temperature=0.0,  # Strict factual grounding
                messages=messages
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ❌ Groq LLM generation failed: {e}")
            answer = "ERROR: Generation Failed"

        latency = time.time() - start_time
        
        questions.append(question)
        ground_truths.append(ground_truth)
        responses.append(answer)
        latencies.append(latency)
        
        print(f"  -> Generated answer in {latency:.2f}s.")

    # Create local evaluation dataframe
    eval_df = pd.DataFrame({
        "question": questions,
        "ground_truth": ground_truths,
        "answer": responses,
        "contexts": retrieved_contexts,
        "latency": latencies
    })

    # --- 7. Run Ragas Metrics Assessment ---
    print("\nPreparing evaluation dataset for Ragas...")
    
    # Ragas expects: 'question' (str), 'answer' (str), 'contexts' (list of str), 'ground_truth' (str)
    # We also pass 'retrieved_contexts' and 'ground_truths' to stay perfectly aligned with cell 424
    eval_df['retrieved_contexts'] = eval_df['contexts']
    
    ragas_dataset = Dataset.from_pandas(eval_df)

    if not openai_api_key:
        print("⚠️ Warning: OPENAI_API_KEY environment variable is not defined.")
        print("⚠️ Ragas evaluation step requires an LLM judge. Attempting run but it may fail if keys are missing.")

    print("📊 Executing Ragas metrics...")
    try:
        eval_result = evaluate(
            ragas_dataset,
            metrics=[
                answer_correctness,
                context_recall,
                faithfulness,
                answer_satisfaction
            ]
        )
        scores = dict(eval_result._repr_dict.items())
        print("\n✅ Ragas Metrics Results:")
        for metric, score in scores.items():
            print(f"  - {metric}: {score:.4f}")
    except Exception as e:
        print(f"❌ Ragas evaluation failed: {e}")
        scores = {}

    # --- 8. Save Reports ---
    os.makedirs("reports", exist_ok=True)
    
    # Save raw answers and scores as JSON
    report_data = {
        "metadata": {
            "device": device,
            "groq_model": config.GROQ_MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
            "timestamp": time.time(),
            "average_latency_sec": sum(latencies)/len(latencies) if latencies else 0
        },
        "scores": scores,
        "records": eval_df.to_dict(orient="records")
    }
    
    json_path = "reports/benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, default=str)
    print(f"\n📁 Saved raw JSON report to: {json_path}")
    
    # Save a clean markdown summary report
    md_path = "reports/benchmark_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Legal AI RAG Benchmark Report\n\n")
        f.write("## Metadata\n")
        f.write(f"- **Groq LLM**: `{config.GROQ_MODEL}`\n")
        f.write(f"- **Embedding Model**: `{config.EMBEDDING_MODEL}`\n")
        f.write(f"- **Average Latency**: `{sum(latencies)/len(latencies):.2f} seconds`\n\n")
        
        if scores:
            f.write("## Evaluation Scores (Ragas)\n")
            for metric, score in scores.items():
                f.write(f"- **{metric.replace('_', ' ').title()}**: `{score:.4f}`\n")
            f.write("\n")
            
        f.write("## Detailed Evaluation Runs\n")
        f.write("| # | Question | Ground Truth | Response | Latency |\n")
        f.write("|---|---|---|---|---|\n")
        for idx, row in eval_df.iterrows():
            q_clean = row['question'].replace('\n', ' ')[:50] + '...' if len(row['question']) > 50 else row['question']
            g_clean = str(row['ground_truth']).replace('\n', ' ')[:50] + '...' if len(str(row['ground_truth'])) > 50 else str(row['ground_truth'])
            a_clean = str(row['answer']).replace('\n', ' ')[:50] + '...' if len(str(row['answer'])) > 50 else str(row['answer'])
            f.write(f"| {idx+1} | {q_clean} | {g_clean} | {a_clean} | {row['latency']:.2f}s |\n")
            
    print(f"📁 Saved Markdown summary report to: {md_path}")
    print("\nBenchmark Execution Complete!")

if __name__ == "__main__":
    run_evaluation()
