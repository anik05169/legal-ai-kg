from rag_pipeline import LegalGraphRAG

engine = LegalGraphRAG()
query = "What are the key conditions of the contract?"
query_embedding = engine.embedder.encode([query]).tolist()
results = engine.collection.query(
    query_embeddings=query_embedding,
    n_results=5,
    include=["metadatas"]
)
print("Retrieved Chunks:")
retrieved = set()
for md in results['metadatas'][0]:
    retrieved.add(md['chunk_id'])
    print(md['chunk_id'])

print("All edge chunks:")
edge_chunks = set(data.get('source_chunk') for u, v, data in engine.G.edges(data=True))
print(edge_chunks)

print("Intersection:", retrieved.intersection(edge_chunks))
