import sys
from core.kg_indexer import build_infrastructure
from core.visualize_kg import generate_interactive_graph
from core.rag_pipeline import LegalGraphRAG

def main():
    print("==================================================")
    print(" 🏛️  LEGAL AI GRAPH-RAG ORCHESTRATOR  🏛️")
    print("==================================================")
    
    # 1. Build Infrastructure (Vector DB + Knowledge Graph)
    print("\n[STEP 1/3] Building Infrastructure (VectorDB + KG)...")
    build_infrastructure()
    
    # 2. Visualize Graph
    print("\n[STEP 2/3] Generating Knowledge Graph Visualization...")
    generate_interactive_graph()
    
    # 3. Start QA Engine
    print("\n[STEP 3/3] Starting GraphRAG Engine...")
    engine = LegalGraphRAG()
    
    print("\n✅ System Ready! Ask your questions below (type 'exit' or 'quit' to stop).")
    
    while True:
        try:
            query = input("\n🤔 Enter your question: ").strip()
            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            if not query:
                continue
                
            print("\n⚙️ Processing query...")
            answer, contexts, triplets = engine.answer_query(query)
            
            print("\n" + "="*60)
            print("🧩 [EXTRACTED KNOWLEDGE GRAPH TRIPLETS]")
            if triplets:
                for t in triplets:
                    print(f"  - {t}")
            else:
                print("  (No relevant triplets found)")
            
            print("\n📄 [PULLED CONTEXT EXCERPTS]")
            if contexts:
                for i, c in enumerate(contexts, 1):
                    print(f"\n--- Excerpt {i} ---\n{c.strip()}")
            else:
                print("  (No relevant context found)")
                
            print("\n🤖 [AI ASSISTANT ANSWER]")
            print(answer)
            print("="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
