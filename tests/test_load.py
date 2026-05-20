print("1. Loading SentenceTransformer...")
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("✅ Embedder loaded.")

print("2. Loading GLiNER...")
from gliner import GLiNER
ner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
print("✅ GLiNER loaded.")

print("3. Loading GLiREL...")
from glirel import GLiREL
re_model = GLiREL.from_pretrained("knowledgator/glirel-base-v1.0")
print("✅ GLiREL loaded.")

print("All models loaded successfully into memory!")