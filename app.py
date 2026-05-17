from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import os
import json

from kg_indexer import build_infrastructure
from visualize_kg import generate_interactive_graph
from rag_pipeline import LegalGraphRAG
from data_loader import get_cuad_contracts

app = FastAPI(title="Legal AI GraphRAG API")

# Lazy-loaded engine
engine = None

class ChatRequest(BaseModel):
    query: str

@app.get("/api/build/stream")
def build_pipeline_stream():
    def event_generator():
        try:
            # 1. Build infrastructure
            for update in build_infrastructure():
                yield f"data: {json.dumps(update)}\n\n"
            
            yield f"data: {json.dumps({'status': 'progress', 'message': '🎨 Generating interactive graph...'})}\n\n"
            generate_interactive_graph()
            
            yield f"data: {json.dumps({'status': 'progress', 'message': '🚀 Initializing GraphRAG Engine...'})}\n\n"
            global engine
            engine = LegalGraphRAG()
            
            # Fetch the contract text to send to frontend
            contract_text = get_cuad_contracts(num_samples=1)[0]
            
            yield f"data: {json.dumps({'status': 'complete', 'message': '✅ Build complete!', 'contract': contract_text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/chat")
def chat(request: ChatRequest):
    global engine
    if not engine:
        raise HTTPException(status_code=400, detail="Engine not initialized. Run the build pipeline first.")
    
    try:
        answer, contexts, triplets = engine.answer_query(request.query)
        return {
            "answer": answer,
            "contexts": contexts,
            "triplets": triplets
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph.html")
def get_graph():
    if os.path.exists("interactive_graph.html"):
        return FileResponse("interactive_graph.html")
    raise HTTPException(status_code=404, detail="Graph not generated yet.")

# Mount the static directory to serve index.html, style.css, script.js
# We use html=True to automatically serve index.html at the root (/)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Make sure static directory exists
    os.makedirs("static", exist_ok=True)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
