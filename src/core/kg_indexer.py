import json
import networkx as nx
import os
import shutil
from pymongo import MongoClient
import certifi

from . import config

def build_infrastructure():
    yield {"status": "progress", "message": "🚀 Starting dynamic Knowledge Graph indexing from contract_data.json..."}
    
    # --- 1. Clean Slate for local fallback files (optional) ---
    if os.path.exists(config.VECTOR_DB_DIR):
        print(f"🧹 Clearing old vector database directory at {config.VECTOR_DB_DIR}...")
        try:
            shutil.rmtree(config.VECTOR_DB_DIR)
        except Exception as e:
            print(f"Warning: Could not clear local vector directory: {e}")

    # Create networkx DiGraph
    G = nx.DiGraph()

    # --- 2. Load pre-parsed contract_data.json graph ---
    contract_data_path = os.path.join("research", "import-graph", "contract_data.json")
    if not os.path.exists(contract_data_path):
        # Fallback to absolute or relative directories
        contract_data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "research", "import-graph", "contract_data.json")
        )

    if not os.path.exists(contract_data_path):
        raise FileNotFoundError(f"Could not find contract_data.json at {contract_data_path}")

    yield {"status": "progress", "message": "📖 Loading contract_data.json..."}
    with open(contract_data_path, "r", encoding="utf-8") as f:
        contracts = json.load(f)

    print(f"📖 Loaded {len(contracts)} contracts from {contract_data_path}.")
    yield {"status": "progress", "message": f"📊 Loaded {len(contracts)} contracts. Reconstructing graph elements..."}

    for idx, contract in enumerate(contracts):
        contract_id = contract.get("file_id")
        if not contract_id:
            continue

        # A. Create Contract Node
        G.add_node(
            contract_id,
            entity_type="Contract",
            summary=contract.get("summary", ""),
            contract_type=contract.get("contract_type", ""),
            effective_date=contract.get("effective_date", ""),
            contract_scope=contract.get("contract_scope", ""),
            duration=contract.get("duration", ""),
            end_date=contract.get("end_date", ""),
            total_amount=contract.get("total_amount", "")
        )

        # B. Governing Law Node
        gov_law = contract.get("governing_law")
        if gov_law:
            gov_law_id = f"Location_{gov_law.get('country') or ''}_{gov_law.get('state') or ''}_{gov_law.get('city') or ''}".strip('_')
            G.add_node(
                gov_law_id,
                entity_type="Location",
                address=gov_law.get("address"),
                city=gov_law.get("city"),
                state=gov_law.get("state"),
                country=gov_law.get("country")
            )
            # Edge: Contract -> HAS_GOVERNING_LAW -> Location
            G.add_edge(contract_id, gov_law_id, label="HAS_GOVERNING_LAW", source_chunks=[contract_id])

        # C. Parties Nodes & Edges
        parties = contract.get("parties", [])
        for party in parties:
            party_name = party.get("name")
            if not party_name:
                continue

            G.add_node(party_name, entity_type="Party")

            # Edge: Party -> PARTY_TO -> Contract
            G.add_edge(party_name, contract_id, label="PARTY_TO", role=party.get("role", "Party"), source_chunks=[contract_id])

            # Party Location
            p_loc = party.get("location")
            if p_loc:
                p_loc_id = f"Location_{p_loc.get('country') or ''}_{p_loc.get('state') or ''}_{p_loc.get('city') or ''}".strip('_')
                G.add_node(
                    p_loc_id,
                    entity_type="Location",
                    address=p_loc.get("address"),
                    city=p_loc.get("city"),
                    state=p_loc.get("state"),
                    country=p_loc.get("country")
                )
                # Edge: Party -> HAS_LOCATION -> Location
                G.add_edge(party_name, p_loc_id, label="HAS_LOCATION", source_chunks=[contract_id])

        # D. Clause Nodes & Edges
        clauses = contract.get("clauses", [])
        for clause in clauses:
            clause_type = clause.get("clause_type")
            if not clause_type:
                continue

            clause_id = f"Clause_{contract_id}_{clause_type}"
            G.add_node(
                clause_id,
                entity_type="Clause",
                clause_type=clause_type,
                summary=clause.get("summary", "")
            )
            # Edge: Contract -> HAS_CLAUSE -> Clause
            G.add_edge(contract_id, clause_id, label="HAS_CLAUSE", source_chunks=[contract_id])

    # --- 3. Save local JSON Fallback file ---
    with open(config.GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(G), f, indent=4)
    print(f"\n✅ Local fallback Knowledge Graph saved to {config.GRAPH_PATH}")

    # --- 4. Upload to MongoDB Atlas ---
    mongo_uri = getattr(config, "MONGO_URI", None)
    if mongo_uri:
        try:
            print("💾 Uploading structured Knowledge Graph to MongoDB Atlas...")
            yield {"status": "progress", "message": "💾 Uploading structured Knowledge Graph to MongoDB Atlas..."}
            client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
            db = client["legal_rag"]

            # Clear collections
            db["kg_nodes"].drop()
            db["kg_edges"].drop()

            # Format and format nodes
            nodes_list = []
            for node, data in G.nodes(data=True):
                doc = {"_id": node}
                doc.update(data)
                nodes_list.append(doc)

            # Format and format edges
            edges_list = []
            for u, v, data in G.edges(data=True):
                doc = {
                    "source": u,
                    "target": v,
                    "label": data.get("label"),
                    "source_chunks": data.get("source_chunks", [])
                }
                # Add extra fields (e.g. role for PARTY_TO)
                for k, val in data.items():
                    if k not in ["label", "source_chunks"]:
                        doc[k] = val
                edges_list.append(doc)

            # Bulk insert
            if nodes_list:
                db["kg_nodes"].insert_many(nodes_list)
            if edges_list:
                db["kg_edges"].insert_many(edges_list)

            # Build database indexes
            db["kg_edges"].create_index("source")
            db["kg_edges"].create_index("target")
            db["kg_edges"].create_index("source_chunks")
            
            print(f"✅ Successfully indexed Knowledge Graph to MongoDB Atlas! Nodes: {len(nodes_list)} | Edges: {len(edges_list)}")
            yield {"status": "progress", "message": f"✅ Indexed to Atlas! Nodes: {len(nodes_list)} | Edges: {len(edges_list)}"}
        except Exception as e:
            print(f"❌ Failed to save Knowledge Graph to MongoDB Atlas: {e}")
            yield {"status": "progress", "message": f"⚠️ Atlas index failed: {e}"}
    else:
        print("⚠️ Warning: MONGO_URI is missing. Knowledge Graph stored locally only.")

    print(f"\n✅ Indexing Complete! Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    yield {"status": "progress", "message": f"✅ Indexing Complete! Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}"}

if __name__ == "__main__":
    for step in build_infrastructure():
        pass