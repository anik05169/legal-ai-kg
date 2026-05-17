import json
import networkx as nx
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from openai import OpenAI

import config
from data_loader import get_cuad_contracts
from kg_indexer import build_infrastructure

class LegalGraphRAG:
    def __init__(self):
        print("🚀 Booting Legal GraphRAG Engine...")
        
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL, device=self.device)
        
        self.chroma_client = chromadb.PersistentClient(path=config.VECTOR_DB_DIR)
        # This will create an empty collection instead of crashing if it's missing
        self.collection = self.chroma_client.get_or_create_collection(name="legal_chunks")      
        
        with open(config.GRAPH_PATH, "r", encoding="utf-8") as f:
            self.G = nx.node_link_graph(json.load(f))
            
        self.llm_client = OpenAI(api_key=config.OPENAI_API_KEY)
        print("✅ Engine Ready!\n")

    def answer_query(self, query):
        # --- 1. SEMANTIC VECTOR SEARCH ---
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=5, # Top 4 chunks is usually a sweet spot
            include=["metadatas", "documents"]
        )
        
        retrieved_texts = []
        retrieved_chunk_ids = set()
        
        if results['metadatas'] and len(results['metadatas'][0]) > 0:
            for i in range(len(results['metadatas'][0])):
                doc_text = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                chunk_id = metadata.get("chunk_id", metadata.get("parent_id"))
                
                retrieved_texts.append(doc_text)
                retrieved_chunk_ids.add(chunk_id)
        else:
            return "No relevant context found.", [], []

        # --- 2. QUERY-TO-GRAPH MATCHING ---
        query_lower = query.lower()
        matched_nodes = set()
        
        # Heuristic: If user asks about dates, pull in Date/Year nodes
        is_date_query = any(w in query_lower for w in ["when", "date", "signed", "terminated", "term", "year", "time"])
        
        for node, data in self.G.nodes(data=True):
            if isinstance(node, str) and len(node) > 4 and node.lower() in query_lower:
                matched_nodes.add(node)
            
            # Boost Date nodes for date-related queries
            if is_date_query and data.get("entity_type") in ["Date", "Year", "Notice Period"]:
                matched_nodes.add(node)

        # --- 3. GRAPH EXPANSION (1-HOP) ---
        expanded_chunk_ids = set()
        extracted_triples = set()
        
        for u, v, data in self.G.edges(data=True):
            source_chunks = data.get("source_chunks", [])
            # Fallback for old index compatibility
            if "source_chunk" in data and data["source_chunk"] not in source_chunks:
                source_chunks.append(data["source_chunk"])
                
            is_in_retrieved = any(chunk in retrieved_chunk_ids for chunk in source_chunks)
            
            # Include edge if it belongs to a semantic chunk OR connects to a matched query entity
            if is_in_retrieved or u in matched_nodes or v in matched_nodes:
                u_type = self.G.nodes[u].get('entity_type', 'Entity')
                v_type = self.G.nodes[v].get('entity_type', 'Entity')
                extracted_triples.add(f"[{u} ({u_type})] --({data['label']})--> [{v} ({v_type})]")
                
                # Hopping: Collect all chunks that contain this structurally relevant edge
                for chunk in source_chunks:
                    expanded_chunk_ids.add(chunk)

        # --- 4. SECONDARY GRAPH RETRIEVAL ---
        # Fetch chunks discovered via the Graph that weren't in our Semantic Search
        # We limit to 3 to prevent overloading the LLM context window
        new_chunk_ids = list(expanded_chunk_ids - retrieved_chunk_ids)[:4]
        graph_expanded_texts = []
        
        if new_chunk_ids:
            graph_results = self.collection.get(ids=new_chunk_ids)
            if graph_results and graph_results['documents']:
                graph_expanded_texts = graph_results['documents']

        # --- 5. CONTEXT SYNTHESIS ---
        context_blocks = []
        if extracted_triples:
            context_blocks.append("=== STRUCTURED KNOWLEDGE GRAPH FACTS ===\n" + "\n".join(extracted_triples))
            
        if retrieved_texts:
            context_blocks.append("=== SEMANTIC MATCHES ===\n" + "\n---\n".join(retrieved_texts))
            
        if graph_expanded_texts:
            context_blocks.append("=== GRAPH-EXPANDED LOGICAL CONNECTIONS ===\n" + "\n---\n".join(graph_expanded_texts))
            
        final_context = "\n\n".join(context_blocks)
        
        # --- 4. LLM GENERATION (Natural, Conversational Prompt) ---
        system_prompt = (
            "You are an expert legal AI assistant. Your job is to answer the user's question clearly "
            "and naturally using ONLY the provided context.\n"
            "RULES:\n"
            "1. Read the provided KNOWLEDGE GRAPH FACTS and LEGAL TEXT EXCERPTS.\n"
            "2. Answer in a helpful, conversational tone. You may explain the context around the answer.\n"
            "3. If the answer is not in the text, simply state that the provided document does not contain the answer.\n"
            "4. Do not provide external legal advice or hallucinate outside knowledge."
        )
        
        response = self.llm_client.chat.completions.create(
            model=config.OPENAI_MODEL,
            temperature=0.2, # Slight temperature bump so it sounds natural but stays accurate
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXT:\n{final_context}\n\nQUERY: {query}"}
            ]
        )
        return response.choices[0].message.content.strip(), retrieved_texts + graph_expanded_texts, list(extracted_triples)

if __name__ == "__main__":
    # If run standalone, just initialize the engine and run a test query
    print("⚠️ Running standalone. Ensure the infrastructure is already built.")
    engine = LegalGraphRAG()
    print("\nTest Answer:\n", engine.answer_query("What is the governing law?")[0])