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
from groq import Groq

# --- FORCE IPV4 ONLY (Bypasses broken IPv6 routing) ---
import socket
orig_getaddrinfo = socket.getaddrinfo
def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = patched_getaddrinfo

# Add parent directory to path to ensure configs can be loaded if needed
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.core import config

def main():
    parser = argparse.ArgumentParser(description="Run 22-question Vector RAG pipeline using MongoDB and Groq.")
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
    
    if not mongo_uri:
        print("❌ ERROR: MONGO_URI environment variable or configuration is missing.", flush=True)
        sys.exit(1)
        
    if not groq_api_key:
        print("❌ ERROR: GROQ_API_KEY environment variable or configuration is missing.", flush=True)
        sys.exit(1)
        
    if not os.path.exists(args.csv):
        print(f"❌ ERROR: Benchmark file not found at {args.csv}", flush=True)
        sys.exit(1)

    print("==================================================", flush=True)
    print("     LEGAL RAG - 22 QUESTION PIPELINE GENERATOR")
    print("==================================================", flush=True)
    print(f"Loading questions from: {args.csv}", flush=True)
    
    try:
        test_df = pd.read_csv(args.csv, delimiter=";")
        print(f"Loaded {len(test_df)} questions.", flush=True)
    except Exception as e:
        print(f"❌ Failed to parse CSV: {e}", flush=True)
        sys.exit(1)

    # --- 2. Load Embedding Model ---
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: [{device.upper()}]", flush=True)
    print("Loading embedding model...", flush=True)
    embedder = SentenceTransformer(config.EMBEDDING_MODEL, device=device)

    # --- 3. Connect to MongoDB Atlas ---
    print("Connecting to MongoDB Atlas...", flush=True)
    try:
        mongo_client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
        db = mongo_client["legal_rag"]
        collection = db["chunks"]
        collection.find_one()  # Validate connection
        print("✅ Successfully connected to MongoDB Atlas.", flush=True)
    except Exception as e:
        print(f"❌ MongoDB Connection failed: {e}", flush=True)
        sys.exit(1)

    # --- 3b. Pre-load Knowledge Graph nodes for fast per-question matching ---
    kg_available = False
    all_kg_nodes = []
    kg_node_types = {}
    try:
        node_count = db["kg_nodes"].count_documents({})
        if node_count > 0:
            all_kg_nodes = list(db["kg_nodes"].find({}))
            kg_node_types = {doc["_id"]: doc.get("entity_type", "Entity") for doc in all_kg_nodes}
            kg_available = True
            print(f"📊 Knowledge Graph loaded: {node_count} nodes available for hybrid retrieval.", flush=True)
        else:
            print("⚠️ kg_nodes collection is empty. Running in vector-only mode.", flush=True)
    except Exception as e:
        print(f"⚠️ Could not load Knowledge Graph: {e}. Running in vector-only mode.", flush=True)

    # --- 4. Initialize Groq Client ---
    print("Initializing Groq client...", flush=True)
    llm_client = Groq(api_key=groq_api_key)

    records = []

    # --- 5. Generate Answers Loop ---
    print("\nExecuting RAG loop for all 22 questions...", flush=True)
    for idx, row in test_df.iterrows():
        question = row["question"]
        ground_truth = row["ground_truth"]
        
        print(f"\n[{idx+1}/22] Question: {question}", flush=True)
        
        start_time = time.time()
        
        # A. Vector search context retrieval (capping context to 3 chunks to save tokens)
        query_embedding = embedder.encode([question], normalize_embeddings=True)[0].tolist()
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": 3
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
            print(f"  ❌ MongoDB Search failed: {e}", flush=True)
            context_list = []

        # B. Knowledge Graph lookup (hybrid retrieval)
        extracted_triples = []
        if kg_available:
            try:
                query_lower = question.lower()
                matched_nodes = set()
                is_date_query = any(w in query_lower for w in ["when", "date", "signed", "terminated", "term", "year", "time", "ended", "end"])

                for doc in all_kg_nodes:
                    node = doc["_id"]
                    ent_type = doc.get("entity_type", "Entity")
                    if isinstance(node, str) and len(node) > 4 and node.lower() in query_lower:
                        matched_nodes.add(node)
                    if is_date_query and ent_type in ["Date", "Year", "Notice Period"]:
                        matched_nodes.add(node)

                if matched_nodes:
                    query_filter = {
                        "$or": [
                            {"source": {"$in": list(matched_nodes)}},
                            {"target": {"$in": list(matched_nodes)}}
                        ]
                    }
                    edges = list(db["kg_edges"].find(query_filter).limit(20))
                    for edge in edges:
                        u = edge["source"]
                        v = edge["target"]
                        label = edge.get("label", "RELATED_TO")
                        u_type = kg_node_types.get(u, "Entity")
                        v_type = kg_node_types.get(v, "Entity")
                        extracted_triples.append(f"[{u} ({u_type})] --({label})--> [{v} ({v_type})]")
                    
                    if extracted_triples:
                        print(f"  📊 KG: {len(extracted_triples)} graph triples matched.", flush=True)
            except Exception as e:
                print(f"  ⚠️ KG lookup error: {e}", flush=True)

        # C. Assemble hybrid context
        context_blocks = []
        if extracted_triples:
            context_blocks.append("=== STRUCTURED KNOWLEDGE GRAPH FACTS ===\n" + "\n".join(extracted_triples))
        if context_list:
            context_blocks.append("=== LEGAL TEXT EXCERPTS ===\n" + "\n---\n".join(context_list))
        
        final_context = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."
        
        system_prompt = (
            "You are an expert legal AI assistant. Your job is to answer the user's question clearly "
            "and naturally using ONLY the provided context.\n"
            "RULES:\n"
            "1. Read the provided KNOWLEDGE GRAPH FACTS and LEGAL TEXT EXCERPTS.\n"
            "2. Answer in a helpful, conversational tone.\n"
            "3. If the answer is not in the text, simply state that the provided document does not contain the answer.\n"
            "4. Do not provide external legal advice or hallucinate outside knowledge."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n{final_context}\n\nQUERY: {question}"}
        ]

        # Use llama-3.1-8b-instant for fast generation and large rate limit capacity
        model_name = "llama-3.1-8b-instant"

        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                temperature=0.0,
                messages=messages
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ❌ Groq Generation failed: {e}", flush=True)
            answer = f"ERROR: Generation failed. {e}"

        latency = time.time() - start_time
        print(f"  -> Generated answer in {latency:.2f}s.", flush=True)
        print(f"  -> Response: {answer[:120]}...", flush=True)
        
        records.append({
            "idx": idx+1,
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "latency": latency
        })
        
        # 1.0 second delay to safeguard API limits
        time.sleep(1.0)

    # --- 6. Save Reports ---
    os.makedirs("reports", exist_ok=True)
    
    # Save raw JSON
    json_path = "reports/rag_answers.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, default=str)
    print(f"\n📁 Saved raw JSON answers to: {json_path}", flush=True)

    # Save a clean summary Markdown report
    md_path = "reports/rag_answers.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# RAG Pipeline System Answers\n\n")
        f.write("This report displays the generated answers for the 22 database benchmark questions.\n\n")
        f.write("## Metadata\n")
        f.write(f"- **Generation LLM**: `llama-3.1-8b-instant`\n")
        f.write(f"- **Vector Index**: MongoDB Atlas (`legal_rag.chunks`)\n")
        f.write(f"- **Knowledge Graph**: MongoDB Atlas (`legal_rag.kg_nodes` + `legal_rag.kg_edges`)\n")
        f.write(f"- **Retrieval Mode**: Hybrid (Vector + Graph)\n")
        f.write(f"- **Total Questions**: {len(records)}\n\n")
        
        f.write("## Questions & Answers\n\n")
        for rec in records:
            f.write(f"### Q{rec['idx']}: {rec['question']}\n")
            f.write(f"- **Expected (Ground Truth)**: {rec['ground_truth']}\n")
            f.write(f"- **AI System Response**: {rec['answer']}\n")
            f.write(f"- **Response Latency**: `{rec['latency']:.2f} seconds`\n\n")
            f.write("---\n\n")
            
    print(f"📁 Saved Markdown answers report to: {md_path}", flush=True)
    print("\nSystem Answer Extraction Complete!", flush=True)

if __name__ == "__main__":
    main()
