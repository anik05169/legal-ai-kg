import chromadb
import json
import networkx as nx

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="legal_chunks")
results = collection.get(include=["metadatas"])
print("Metadata from Chroma:", results["metadatas"])

with open("legal_kg.json", "r") as f:
    G = nx.node_link_graph(json.load(f))
print("Edges from Graph:", list(G.edges(data=True))[:2])
