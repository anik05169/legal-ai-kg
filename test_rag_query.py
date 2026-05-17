from rag_pipeline import LegalGraphRAG

engine = LegalGraphRAG()
query = "What are the key conditions of the contract?"
answer, contexts, triplets = engine.answer_query(query)
print("Triplets:", triplets)
