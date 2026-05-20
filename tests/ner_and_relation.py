from gliner import GLiNER
from glirel import GLiREL
import spacy

# =========================
# Load Models
# =========================
print("Loading GLiNER...")
ner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

print("Loading GLiREL...")
rel_model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")

print("Loading spaCy...")
nlp = spacy.load("en_core_web_sm")

# =========================
# Define Ontology & Rules
# =========================
entity_labels = ["Person", "Organization", "Location", "Title"]
relation_labels = ["works_for", "ceo_of", "located_in", "founded_by"]

# Added bi-directional rules to catch reversed relations
VALID_RELATIONS = {
    ("Person", "Organization"): ["works_for", "ceo_of", "founded_by"],
    ("Organization", "Person"): ["works_for", "ceo_of", "founded_by"],
    ("Organization", "Location"): ["located_in"],
    ("Location", "Organization"): ["located_in"]
}

# =========================
# Input Text & Preprocessing
# =========================
text = "Tim Cook is the CEO of Apple Inc., which is headquartered in Cupertino."
print(f"\nINPUT TEXT:\n{text}")

doc = nlp(text)
tokens = [token.text for token in doc]
entities = ner_model.predict_entities(text, entity_labels, threshold=0.5)

# =========================
# Step 1: Prepare Entities
# =========================
glirel_ner = []
entity_lookup = {}

print("\n--- Extracted NER Spans ---")
for ent in entities:
    span = doc.char_span(ent["start"], ent["end"], alignment_mode="expand")
    if span:
        start_idx = span.start
        end_idx = span.end - 1  # Inclusive bounds
        
        glirel_ner.append([start_idx, end_idx, ent["label"]])
        entity_lookup[(start_idx, end_idx)] = ent["label"]
        print(f"{ent['text']} ({ent['label']}) -> tokens [{start_idx}, {end_idx}]")

# =========================
# Step 2: Extract Relations
# =========================
relations = rel_model.predict_relations(
    tokens,
    labels=relation_labels,
    threshold=0.45,
    ner=glirel_ner,
    top_k=1
)

print("\n--- Raw Relations ---")
for rel in relations:
    head = " ".join(rel["head_text"])
    tail = " ".join(rel["tail_text"])
    print(f"{head} --[{rel['label']}]--> {tail} (score={rel['score']:.3f})")

# =========================
# Step 3: Robust Filtering
# =========================
# Helper function to detect entities even if GLiREL alters the span slightly
def get_entity_type(target_span, lookup):
    if target_span in lookup:
        return lookup[target_span]
    
    # Fallback: Find overlapping spans
    for (s, e), label in lookup.items():
        if max(target_span[0], s) <= min(target_span[1], e):
            return label
    return None

filtered_relations = []

for rel in relations:
    head_span = tuple(rel["head_pos"])
    tail_span = tuple(rel["tail_pos"])

    # Use the robust lookup
    head_type = get_entity_type(head_span, entity_lookup)
    tail_type = get_entity_type(tail_span, entity_lookup)

    allowed = VALID_RELATIONS.get((head_type, tail_type), [])

    if rel["label"] in allowed:
        filtered_relations.append(rel)

# =========================
# Step 4: Final Output
# =========================
print("\n--- Final Relations ---")

if not filtered_relations:
    print("No valid relations found.")
else:
    for rel in filtered_relations:
        head = " ".join(rel["head_text"]).replace(" ,", ",").replace(" .", ".")
        tail = " ".join(rel["tail_text"]).replace(" ,", ",").replace(" .", ".")
        print(f"{head} --[{rel['label']}]--> {tail} (score={rel['score']:.3f})")