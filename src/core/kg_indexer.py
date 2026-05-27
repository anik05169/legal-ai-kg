import json
import networkx as nx
import os
import shutil
from pymongo import MongoClient
import certifi

from . import config

# ==========================================
# 🛠️ HELPER: Safe string conversion
# ==========================================
def _safe_str(val):
    """Convert None to empty string, everything else to str."""
    if val is None:
        return ""
    return str(val)


def _make_location_id(loc_dict):
    """
    Build a deterministic, collision-free Location node ID.
    Handles null/missing fields gracefully so two parties in
    different contracts with country="US" and everything else
    null still merge into the same Location node (correct behavior).
    """
    parts = [
        _safe_str(loc_dict.get("country")),
        _safe_str(loc_dict.get("state")),
        _safe_str(loc_dict.get("city")),
        _safe_str(loc_dict.get("address")),
    ]
    # Filter out empties, join with underscore
    suffix = "_".join(p for p in parts if p)
    return f"Location_{suffix}" if suffix else None


# ==========================================
# 🚀 MAIN PIPELINE
# ==========================================
def build_infrastructure():
    yield {"status": "progress", "message": "🚀 Starting Knowledge Graph indexing from contract_data.json..."}

    # --- 1. Clean Slate for local fallback files (optional) ---
    if os.path.exists(config.VECTOR_DB_DIR):
        print(f"🧹 Clearing old vector database directory at {config.VECTOR_DB_DIR}...", flush=True)
        try:
            shutil.rmtree(config.VECTOR_DB_DIR)
        except Exception as e:
            print(f"Warning: Could not clear local vector directory: {e}", flush=True)

    G = nx.DiGraph()

    # --- 2. Locate and load contract_data.json ---
    # Try repo-root-relative path first (works in CI), then __file__-relative (works locally)
    contract_data_path = os.path.join("research", "import-graph", "contract_data.json")
    if not os.path.exists(contract_data_path):
        contract_data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "research", "import-graph", "contract_data.json")
        )

    if not os.path.exists(contract_data_path):
        msg = f"Could not find contract_data.json (tried CWD and __file__ relative paths)"
        print(f"❌ {msg}", flush=True)
        raise FileNotFoundError(msg)

    yield {"status": "progress", "message": "📖 Loading contract_data.json..."}
    print(f"📖 Loading {contract_data_path} ...", flush=True)
    with open(contract_data_path, "r", encoding="utf-8") as f:
        contracts = json.load(f)

    total = len(contracts)
    print(f"📊 Loaded {total} contracts. Building graph...", flush=True)
    yield {"status": "progress", "message": f"📊 Loaded {total} contracts. Reconstructing graph elements..."}

    skipped = 0
    for idx, contract in enumerate(contracts):
        contract_id = contract.get("file_id")
        if not contract_id:
            skipped += 1
            continue

        # Progress every 100 contracts
        if (idx + 1) % 100 == 0 or idx == total - 1:
            print(f"  Processing contract {idx + 1}/{total}...", flush=True)

        # A. Contract Node
        G.add_node(
            contract_id,
            entity_type="Contract",
            summary=_safe_str(contract.get("summary")),
            contract_type=_safe_str(contract.get("contract_type")),
            effective_date=_safe_str(contract.get("effective_date")),
            contract_scope=_safe_str(contract.get("contract_scope")),
            duration=_safe_str(contract.get("duration")),
            end_date=_safe_str(contract.get("end_date")),
            total_amount=_safe_str(contract.get("total_amount")),
        )

        # B. Governing Law Location
        gov_law = contract.get("governing_law")
        if gov_law and isinstance(gov_law, dict):
            gov_law_id = _make_location_id(gov_law)
            if gov_law_id:
                G.add_node(
                    gov_law_id,
                    entity_type="Location",
                    address=_safe_str(gov_law.get("address")),
                    city=_safe_str(gov_law.get("city")),
                    state=_safe_str(gov_law.get("state")),
                    country=_safe_str(gov_law.get("country")),
                )
                G.add_edge(contract_id, gov_law_id, label="HAS_GOVERNING_LAW", source_chunks=[contract_id])

        # C. Party Nodes & Edges
        for party in contract.get("parties", []):
            party_name = party.get("name")
            if not party_name:
                continue

            G.add_node(party_name, entity_type="Party")
            G.add_edge(
                party_name, contract_id,
                label="PARTY_TO",
                role=_safe_str(party.get("role", "Party")),
                source_chunks=[contract_id],
            )

            # Party Location
            p_loc = party.get("location")
            if p_loc and isinstance(p_loc, dict):
                p_loc_id = _make_location_id(p_loc)
                if p_loc_id:
                    G.add_node(
                        p_loc_id,
                        entity_type="Location",
                        address=_safe_str(p_loc.get("address")),
                        city=_safe_str(p_loc.get("city")),
                        state=_safe_str(p_loc.get("state")),
                        country=_safe_str(p_loc.get("country")),
                    )
                    G.add_edge(party_name, p_loc_id, label="HAS_LOCATION", source_chunks=[contract_id])

        # D. Clause Nodes & Edges
        for clause in contract.get("clauses", []):
            clause_type = clause.get("clause_type")
            if not clause_type:
                continue

            clause_id = f"Clause_{contract_id}_{clause_type}"
            G.add_node(
                clause_id,
                entity_type="Clause",
                clause_type=_safe_str(clause_type),
                summary=_safe_str(clause.get("summary")),
            )
            G.add_edge(contract_id, clause_id, label="HAS_CLAUSE", source_chunks=[contract_id])

    if skipped:
        print(f"⚠️ Skipped {skipped} contracts with missing file_id.", flush=True)

    print(f"✅ Graph built: {G.number_of_nodes()} nodes | {G.number_of_edges()} edges", flush=True)

    # --- 3. Save local JSON fallback ---
    try:
        with open(config.GRAPH_PATH, "w", encoding="utf-8") as f:
            json.dump(nx.node_link_data(G), f, indent=2)
        print(f"💾 Local fallback saved to {config.GRAPH_PATH}", flush=True)
    except Exception as e:
        print(f"⚠️ Could not save local fallback: {e}", flush=True)

    # --- 4. Upload to MongoDB Atlas ---
    mongo_uri = getattr(config, "MONGO_URI", None) or os.getenv("MONGO_URI", "")
    mongo_uri = mongo_uri.strip() if mongo_uri else ""

    if not mongo_uri:
        print("⚠️ MONGO_URI not set — Knowledge Graph stored locally only.", flush=True)
        yield {"status": "progress", "message": "⚠️ MONGO_URI not set. Local-only mode."}
    else:
        try:
            print("💾 Connecting to MongoDB Atlas...", flush=True)
            yield {"status": "progress", "message": "💾 Uploading Knowledge Graph to MongoDB Atlas..."}
            client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
            db = client["legal_rag"]

            # Drop old collections
            db["kg_nodes"].drop()
            db["kg_edges"].drop()

            # --- Format nodes ---
            nodes_list = []
            for node, data in G.nodes(data=True):
                doc = {"_id": node}
                doc.update(data)
                nodes_list.append(doc)

            # --- Format edges ---
            edges_list = []
            for u, v, data in G.edges(data=True):
                doc = {
                    "source": u,
                    "target": v,
                    "label": data.get("label"),
                    "source_chunks": data.get("source_chunks", []),
                }
                for k, val in data.items():
                    if k not in ("label", "source_chunks"):
                        doc[k] = val
                edges_list.append(doc)

            # --- Batched insert (safe for large graphs) ---
            BATCH = 5000
            if nodes_list:
                for i in range(0, len(nodes_list), BATCH):
                    batch = nodes_list[i : i + BATCH]
                    db["kg_nodes"].insert_many(batch, ordered=False)
                    print(f"  Inserted nodes batch {i // BATCH + 1} ({len(batch)} docs)", flush=True)

            if edges_list:
                for i in range(0, len(edges_list), BATCH):
                    batch = edges_list[i : i + BATCH]
                    db["kg_edges"].insert_many(batch, ordered=False)
                    print(f"  Inserted edges batch {i // BATCH + 1} ({len(batch)} docs)", flush=True)

            # Build indexes after insert for best performance
            db["kg_edges"].create_index("source")
            db["kg_edges"].create_index("target")
            db["kg_edges"].create_index("source_chunks")
            db["kg_edges"].create_index("label")

            print(f"✅ MongoDB Atlas indexed! Nodes: {len(nodes_list)} | Edges: {len(edges_list)}", flush=True)
            yield {"status": "progress", "message": f"✅ Atlas indexed! Nodes: {len(nodes_list)} | Edges: {len(edges_list)}"}
        except Exception as e:
            print(f"❌ MongoDB Atlas upload failed: {e}", flush=True)
            yield {"status": "progress", "message": f"❌ Atlas upload failed: {e}"}

    print(f"\n✅ Indexing Complete! Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}", flush=True)
    yield {"status": "progress", "message": f"✅ Indexing Complete! Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}"}


if __name__ == "__main__":
    for step in build_infrastructure():
        pass