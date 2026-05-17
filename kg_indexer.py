import json
import uuid
import networkx as nx
import chromadb
import torch
import shutil
import os
from sentence_transformers import SentenceTransformer
from gliner import GLiNER
from glirel import GLiREL
import spacy

import config
from data_loader import get_cuad_contracts

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================
def chunk_text(text, chunk_size=150, overlap=30):
    """
    Reduced to 150 words to stay safely under GLiNER's 384-token limit.
    This prevents the 'Truncation' warnings and data loss.
    """
    words = text.split()
    chunks = []
    if len(words) == 0: return chunks
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def get_entity_type(target_span, lookup):
    if target_span in lookup: return lookup[target_span]
    for (s, e), label in lookup.items():
        if max(target_span[0], s) <= min(target_span[1], e): return label
    return None

# ==========================================
# 🚀 MAIN PIPELINE
# ==========================================
def build_infrastructure():
    yield {"status": "progress", "message": "Starting infrastructure build..."}
    # --- 1. Clean Slate (Wipe old DB to prevent metadata crashes) ---
    if os.path.exists(config.VECTOR_DB_DIR):
        print(f"🧹 Clearing old database at {config.VECTOR_DB_DIR}...")
        yield {"status": "progress", "message": "🧹 Clearing old database..."}
        shutil.rmtree(config.VECTOR_DB_DIR)

    # --- 2. Hardware Acceleration ---
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"⚡ Device: [{device.upper()}]")
    yield {"status": "progress", "message": f"⚡ Device: [{device.upper()}]"}

    yield {"status": "progress", "message": "🧠 Loading GLiNER and GLiREL models... This may take a moment."}

    ner_model = GLiNER.from_pretrained(config.GLINER_MODEL).to(device)
    rel_model = GLiREL.from_pretrained(config.GLIREL_MODEL).to(device)
    embedder = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    nlp = spacy.blank("en") 
    
    chroma_client = chromadb.PersistentClient(path=config.VECTOR_DB_DIR)
    collection = chroma_client.get_or_create_collection(name="legal_chunks")
    G = nx.DiGraph()
    
    # --- 3. Load Data ---
    contracts = get_cuad_contracts(num_samples=1) 

    print(f"\n🧠 Indexing {len(contracts)} contract(s)...")
    yield {"status": "progress", "message": f"📊 Found {len(contracts)} contract(s). Preparing chunks..."}
    
    for doc_idx, text in enumerate(contracts):
        parent_chunks = chunk_text(text, chunk_size=150, overlap=30)
        
        # Batch Vectorize
        print(f"  -> Batch encoding {len(parent_chunks)} chunks...")
        yield {"status": "progress", "message": f"⏳ Contract has {len(parent_chunks)} chunks. Batch encoding embeddings..."}
        batch_embeddings = embedder.encode(parent_chunks).tolist()
        
        for chunk_idx, parent_text in enumerate(parent_chunks):
            print(f"--- Chunk {chunk_idx + 1}/{len(parent_chunks)} ---", end="\r")
            yield {"status": "progress", "message": f"⚙️ Indexing chunk {chunk_idx + 1} / {len(parent_chunks)}..."}
            parent_id = str(uuid.uuid4())
            
            # NER & Relation Extraction
            doc = nlp(parent_text)
            tokens = [token.text for token in doc]
            entities = ner_model.predict_entities(parent_text, config.ENTITY_LABELS, threshold=0.4)
            
            # ... inside your build_infrastructure.py ...
            glirel_ner, entity_lookup = [], {}
            for ent in entities:
                span = doc.char_span(ent["start"], ent["end"], alignment_mode="expand")
                
                # Anti-Hallucination Filter 
                text_clean = ent["text"].strip()
                # 1. Must be longer than 2 characters (drops IT, BY, ON, TO)
                # 2. Must not be a generic pronoun/stopword
                is_valid = len(text_clean) > 2 and text_clean.lower() not in ["the", "this", "that", "they", "any", "all"]
                
                if span and is_valid:
                    start_idx, end_idx = span.start, span.end - 1
                    glirel_ner.append([start_idx, end_idx, ent["label"]])
                    entity_lookup[(start_idx, end_idx)] = ent["label"]

            if glirel_ner:
                relations = rel_model.predict_relations(
                    tokens, labels=config.RELATION_LABELS, threshold=0.3, ner=glirel_ner, top_k=1
                )
                
                for rel in relations:
                    head_type = get_entity_type(tuple(rel["head_pos"]), entity_lookup)
                    tail_type = get_entity_type(tuple(rel["tail_pos"]), entity_lookup)
                    src = " ".join(rel["head_text"]).replace(" ,", ",").strip(".,;: \n\t").upper()
                    tgt = " ".join(rel["tail_text"]).replace(" ,", ",").strip(".,;: \n\t").upper()
                    
                    if src == tgt or not src or not tgt: continue

                    if rel["label"] in config.VALID_RELATIONS.get((head_type, tail_type), []):
                        G.add_node(src, entity_type=head_type)
                        G.add_node(tgt, entity_type=tail_type)
                        
                        if G.has_edge(src, tgt):
                            # Append chunk to existing edge to prevent overwriting
                            existing_chunks = G[src][tgt].get("source_chunks", [])
                            if "source_chunk" in G[src][tgt] and G[src][tgt]["source_chunk"] not in existing_chunks:
                                existing_chunks.append(G[src][tgt]["source_chunk"])
                            if parent_id not in existing_chunks:
                                existing_chunks.append(parent_id)
                            G[src][tgt]["source_chunks"] = existing_chunks
                        else:
                            G.add_edge(src, tgt, label=rel["label"], source_chunks=[parent_id])

            # Save to Vector DB
            collection.add(
                documents=[parent_text],
                embeddings=[batch_embeddings[chunk_idx]],
                metadatas=[{"chunk_id": parent_id}],
                ids=[parent_id]
            )

    # --- 4. Save Graph ---
    with open(config.GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(G), f, indent=4)

    print(f"\n✅ Indexing Complete! Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    yield {"status": "progress", "message": f"✅ Indexing Complete! Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}"}

if __name__ == "__main__":
    for step in build_infrastructure():
        pass