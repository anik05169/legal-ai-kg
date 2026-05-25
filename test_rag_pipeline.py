import os
import sys
import argparse
import json
import certifi
import torch
import networkx as nx
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- FORCE IPV4 ONLY (Bypasses broken IPv6 routing) ---
import socket
orig_getaddrinfo = socket.getaddrinfo
def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = patched_getaddrinfo

# Add parent directory to path to ensure configs can be loaded if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core import config

def main():
    parser = argparse.ArgumentParser(description="Test Legal AI RAG Pipeline with Groq.")
    parser.add_argument(
        "--query", 
        type=str, 
        default="what is breach of contract", 
        help="Query to ask the RAG pipeline"
    )
    args = parser.parse_args()

    # --- 1. Load Credentials & Settings ---
    mongo_uri = os.getenv("MONGO_URI") or getattr(config, "MONGO_URI", None)
    groq_api_key = os.getenv("GROQ_API_KEY") or getattr(config, "GROQ_API_KEY", None)
    
    if not mongo_uri:
        print("❌ ERROR: MONGO_URI environment variable or configuration is missing.")
        sys.exit(1)
        
    if not groq_api_key:
        print("❌ ERROR: GROQ_API_KEY environment variable or configuration is missing.")
        sys.exit(1)

    print("==================================================")
    print(" 🏛️  LEGAL AI — PIPELINE TEST RUNNER (GROQ)  🏛️")
    print("==================================================")
    print(f"🔎 User Query: '{args.query}'\n")

    # --- 2. Hardware Acceleration ---
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"⚡ Device: [{device.upper()}]")

    # --- 3. Load Embedding Model ---
    print("🧠 Loading embedding model...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL, device=device)

    # --- 4. Connect to MongoDB Atlas & Run Vector Search ---
    print("📥 Connecting to MongoDB Atlas...")
    mongo_client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
    db = mongo_client["legal_rag"]
    collection = db["chunks"]

    print("🔍 Fetching semantic chunks...")
    query_embedding = embedder.encode([args.query], normalize_embeddings=True)[0].tolist()

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
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    try:
        retrieved_results = list(collection.aggregate(pipeline))
    except Exception as e:
        print(f"❌ MongoDB Vector Search failed: {e}")
        print("⚠️ Ensure you have created a Vector Search Index named 'vector_index' on your MongoDB collection.")
        sys.exit(1)

    retrieved_texts = []
    retrieved_chunk_ids = set()
    for res in retrieved_results:
        retrieved_texts.append(res["text"])
        # Use chunk_id if present in metadata, otherwise use doc _id
        chunk_id = res.get("metadata", {}).get("chunk_id", res["_id"])
        retrieved_chunk_ids.add(chunk_id)

    print(f"✅ Retrieved {len(retrieved_texts)} semantic chunks from MongoDB.")

    # --- 5. Optional Knowledge Graph Parsing ---
    extracted_triples = []
    kg_path = config.GRAPH_PATH
    if os.path.exists(kg_path):
        print("📊 Loading Knowledge Graph data...")
        try:
            with open(kg_path, "r", encoding="utf-8") as f:
                G = nx.node_link_graph(json.load(f))
                
            query_lower = args.query.lower()
            matched_nodes = set()
            is_date_query = any(w in query_lower for w in ["when", "date", "signed", "terminated", "term", "year", "time"])
            
            for node, data in G.nodes(data=True):
                if isinstance(node, str) and len(node) > 4 and node.lower() in query_lower:
                    matched_nodes.add(node)
                if is_date_query and data.get("entity_type") in ["Date", "Year", "Notice Period"]:
                    matched_nodes.add(node)
                    
            for u, v, data in G.edges(data=True):
                source_chunks = data.get("source_chunks", [])
                if "source_chunk" in data and data["source_chunk"] not in source_chunks:
                    source_chunks.append(data["source_chunk"])
                    
                is_in_retrieved = any(chunk in retrieved_chunk_ids for chunk in source_chunks)
                if is_in_retrieved or u in matched_nodes or v in matched_nodes:
                    u_type = G.nodes[u].get('entity_type', 'Entity')
                    v_type = G.nodes[v].get('entity_type', 'Entity')
                    extracted_triples.append(f"[{u} ({u_type})] --({data['label']})--> [{v} ({v_type})]")
            print(f"✅ Extracted {len(extracted_triples)} relation triples from Knowledge Graph.")
        except Exception as e:
            print(f"⚠️ Error reading/querying Knowledge Graph: {e}")
    else:
        print("💡 Note: No local Knowledge Graph file (legal_kg.json) found. Proceeding with semantic text context only.")

    # --- 6. Synthesize Context ---
    context_blocks = []
    if extracted_triples:
        context_blocks.append("=== STRUCTURED KNOWLEDGE GRAPH FACTS ===\n" + "\n".join(extracted_triples))
    if retrieved_texts:
        context_blocks.append("=== LEGAL TEXT EXCERPTS ===\n" + "\n---\n".join(retrieved_texts))
        
    final_context = "\n\n".join(context_blocks)

    if not final_context:
        print("❌ No relevant context was found in the database. LLM response may be generic.")
        final_context = "No relevant context found."

    # --- 7. Call LLM (Groq) ---
    print(f"🤖 Invoking Groq LLM ({config.GROQ_MODEL})...")
    llm_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key
    )

    system_prompt = (
        "You are an expert legal AI assistant. Your job is to answer the user's question clearly "
        "and naturally using ONLY the provided context.\n"
        "RULES:\n"
        "1. Read the provided KNOWLEDGE GRAPH FACTS and LEGAL TEXT EXCERPTS.\n"
        "2. Answer in a helpful, conversational tone. You may explain the context around the answer.\n"
        "3. If the answer is not in the text, simply state that the provided document does not contain the answer.\n"
        "4. Do not provide external legal advice or hallucinate outside knowledge."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"CONTEXT:\n{final_context}\n\nQUERY: {args.query}"}
    ]

    try:
        response = llm_client.chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=0.2,
            messages=messages
        )
        answer = response.choices[0].message.content.strip()
        
        print("\n" + "="*60)
        print("📄 [CONTEXT HIGHLIGHTS]")
        for idx, text in enumerate(retrieved_texts[:2]):
            print(f"Excerpt #{idx+1}: {text[:150]}...")
            
        print("\n🤖 [AI ASSISTANT RESPONSE via GROQ]")
        print(answer)
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"❌ Groq LLM API Call failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
