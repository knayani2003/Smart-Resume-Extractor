from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from ats_score_ai import retrieve_answers,es
from ats_score_ai import search_data as external_search_data


app = FastAPI(
    title="CV Search API",
    description="API for searching CV documents using two different search methods",
    version="1.0.0"
)

class SearchQuery(BaseModel):
    query: str

class SearchResponse(BaseModel):
    results: Dict[str, Any]
    message: str

@app.post("/search/")
async def search_cvs(request: SearchQuery):
    """
    Search using direct Elasticsearch query method.
    This method uses standard Elasticsearch query to search through indexed CV data.
    """
    try:
        
        if not es.ping():
            raise HTTPException(status_code=500, detail="Failed to connect to Elasticsearch")

        results = external_search_data(request.query)
        if results:
            return SearchResponse(
                results=results,
                message="Search completed successfully"
            )
        else:
            return SearchResponse(
                results={},
                message="No results found"
            )
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector-search/")
async def vector_search_cvs(query: SearchQuery):
    """
    Search using vector similarity and LangChain QA method.
    This method uses semantic search with vector embeddings for more context-aware results.
    """
    try:
        results = retrieve_answers(query.query)
        if results:
            return SearchResponse(
                results=results,
                message="Vector search completed successfully"
            )
        else:
            return SearchResponse(
                results={},
                message="No results found"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Check if the API and Elasticsearch are running.
    """
    try:
        es_status = es.ping()
        return {
            "status": "healthy",
            "elasticsearch": "connected" if es_status else "disconnected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }