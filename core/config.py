import os

# --- CREDENTIALS ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- MODELS ---
GLINER_MODEL = "urchade/gliner_medium-v2.1"
GLIREL_MODEL = "jackboyla/glirel-large-v0"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
OPENAI_MODEL = "gpt-4o-mini" 
REL_THRESHOLD = 0.4
NER_THRESHOLD = 0.6

# --- PATHS ---
VECTOR_DB_DIR = "./chroma_db"
GRAPH_PATH = "./legal_kg.json"

ENTITY_LABELS = [
    "Title", "Date", "Year","Money", "Governing Law", "Jurisdiction", "License Grant", 
    "Payment Term", "Termination Clause", "Limitation of Liability", "Notice Period", "Price",
    "Location", "Person", "Organization"
]

RELATION_LABELS = [
    "entered_into_by", "signed_on", "governed_by", "has_jurisdiction", "terminates_on",
    "agrees_to", "located_in", "part_of", "has_term", "has_price", "is_liable_to", "pays_to"
]

VALID_RELATIONS = {
    # Document Relationships
    ("Document Name", "Party"): ["entered_into_by"],
    ("Party", "Document Name"): ["entered_into_by"],
    ("Document Name", "Organization"): ["entered_into_by"],
    ("Organization", "Document Name"): ["entered_into_by"],
    ("Document Name", "Person"): ["entered_into_by"],
    ("Person", "Document Name"): ["entered_into_by"],
    
    ("Document Name", "Date"): ["signed_on", "terminates_on"],
    ("Date", "Document Name"): ["signed_on", "terminates_on"],
    
    ("Party", "Date"): ["signed_on", "terminates_on"],
    ("Date", "Party"): ["signed_on", "terminates_on"],
    
    ("Organization", "Date"): ["signed_on", "terminates_on"],
    ("Date", "Organization"): ["signed_on", "terminates_on"],
    
    ("Document Name", "Year"): ["signed_on", "terminates_on"],
    ("Year", "Document Name"): ["signed_on", "terminates_on"],
    
    ("Party", "Year"): ["signed_on", "terminates_on"],
    ("Year", "Party"): ["signed_on", "terminates_on"],
    
    ("Document Name", "Governing Law"): ["governed_by"],
    ("Governing Law", "Document Name"): ["governed_by"],
    
    ("Document Name", "Jurisdiction"): ["has_jurisdiction"],
    ("Jurisdiction", "Document Name"): ["has_jurisdiction"],
    
    # Contract Terms
    ("Document Name", "License Grant"): ["has_term"],
    ("License Grant", "Document Name"): ["has_term"],
    
    ("Document Name", "Payment Term"): ["has_term"],
    ("Payment Term", "Document Name"): ["has_term"],
    
    ("Document Name", "Termination Clause"): ["has_term"],
    ("Termination Clause", "Document Name"): ["has_term"],
    
    ("Document Name", "Limitation of Liability"): ["has_term"],
    ("Limitation of Liability", "Document Name"): ["has_term"],
    
    ("Document Name", "Notice Period"): ["has_term"],
    ("Notice Period", "Document Name"): ["has_term"],
    
    ("Document Name", "Price"): ["has_price"],
    ("Price", "Document Name"): ["has_price"],
    ("Payment Term", "Price"): ["has_price"],
    ("Price", "Payment Term"): ["has_price"],
    
    # Inter-Entity Relationships
    ("Party", "Party"): ["agrees_to"],
    ("Organization", "Organization"): ["agrees_to", "part_of"],
    ("Party", "Organization"): ["part_of", "located_in", "agrees_to"],
    ("Organization", "Party"): ["part_of", "located_in", "agrees_to"],
    ("Person", "Organization"): ["part_of", "agrees_to"],
    ("Organization", "Person"): ["part_of", "agrees_to"],
    ("Person", "Party"): ["part_of", "agrees_to"],
    ("Party", "Person"): ["part_of", "agrees_to"],
    
    # Spatial Relationships
    ("Party", "Location"): ["located_in"],
    ("Location", "Party"): ["located_in"],
    ("Organization", "Location"): ["located_in"],
    ("Location", "Organization"): ["located_in"],
    ("Person", "Location"): ["located_in"],
    ("Location", "Person"): ["located_in"],
}