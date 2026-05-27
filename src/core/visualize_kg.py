import json
import networkx as nx
from pyvis.network import Network
from . import config

# Define a beautiful color palette for your legal entities
COLOR_MAP = {
    "Party": "#FF5A5F",             # Coral Red
    "Organization": "#087E8B",      # Teal
    "Document Name": "#FFC857",     # Golden Yellow
    "Location": "#59C9A5",          # Mint Green
    "Date": "#D3D3D3",              # Light Grey
    "Price": "#85C7F2",             # Light Blue
    "Notice Period": "#9B5DE5",     # Purple
    "Governing Law": "#F15BB5",     # Pink
    "Jurisdiction": "#00BBF9"       # Bright Blue
}
DEFAULT_COLOR = "#999999"

def generate_interactive_graph():
    print("🎨 Loading Knowledge Graph for Beautification...")
    
    try:
        with open(config.GRAPH_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            G = nx.node_link_graph(data)
    except FileNotFoundError:
        print("❌ Graph file not found. Please run your indexer first.")
        return

    print(f"📊 Graph contains {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # 1. Calculate Node Importance (Degree) for sizing
    # Nodes with more connections will appear larger
    degrees = dict(G.degree())
    
    # 2. Inject Aesthetics into the NetworkX Graph BEFORE passing to PyVis
    for node_id, node_data in G.nodes(data=True):
        entity_type = node_data.get("entity_type", "Unknown")
        
        # Set Node Color based on Entity Type
        node_data["color"] = COLOR_MAP.get(entity_type, DEFAULT_COLOR)
        
        # Set Node Size based on how many connections it has (Base size 15 + degree * 3)
        node_data["size"] = 15 + (degrees[node_id] * 3)
        
        # Set Hover Text (Tooltip)
        node_data["title"] = f"Type: {entity_type}<br>Connections: {degrees[node_id]}"
        
        # Visual polish
        node_data["borderWidth"] = 2
        node_data["borderWidthSelected"] = 4

    for u, v, edge_data in G.edges(data=True):
        # Set the text that appears on the line itself
        rel_label = edge_data.get("label", "")
        edge_data["title"] = rel_label  # Hover text
        edge_data["label"] = rel_label  # Visible text on the edge
        
        # Edge styling
        edge_data["color"] = "#666666"
        edge_data["width"] = 1.5

    # 3. Initialize PyVis Network
    net = Network(
        height="100vh", # Use full viewport height
        width="100%", 
        directed=True, 
        bgcolor="#1E1E1E", # Sleek dark background
        font_color="#FFFFFF",
        select_menu=True,  # Adds a drop-down to search/select nodes
        filter_menu=True,  # Adds a drop-down to filter by entity type
        cdn_resources="remote"
    )
    
    # Load the styled NetworkX graph
    net.from_nx(G)
    
    # 4. Tune the Physics Engine for a "Beautiful" spread
    # ForceAtlas2Based is generally the best algorithm for Knowledge Graphs
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -100,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08,
          "damping": 0.4
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      },
      "edges": {
        "font": {
          "size": 12,
          "color": "#A0A0A0",
          "align": "top"
        },
        "smooth": {
          "type": "continuous",
          "forceDirection": "none"
        },
        "arrows": {
          "to": { "enabled": true, "scaleFactor": 0.5 }
        }
      }
    }
    """)
    
    output_file = "interactive_graph.html"
    net.write_html(output_file)
    
    print(f"\n✅ Beautiful Visualization generated successfully!")
    print(f"🌐 Open '{output_file}' in your web browser.")

if __name__ == "__main__":
    generate_interactive_graph()