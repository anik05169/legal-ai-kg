import json
import networkx as nx

with open("legal_kg.json", "r") as f:
    G = nx.node_link_graph(json.load(f))

for u, v, data in G.edges(data=True):
    print(f"[{u}] --({data['label']})--> [{v}]")
