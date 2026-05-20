import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.rag_pipeline import LegalGraphRAG

engine = LegalGraphRAG()
query = "What are the key conditions of the contract?"
answer, contexts, triplets = engine.answer_query(query)
print("Triplets:", triplets)
