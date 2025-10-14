"""Query endpoint for VEAH LLM"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

@router.post("/")
async def query(request: QueryRequest):
    """Process a general query"""
    # Model inference logic here
    return {"response": f"Response to: {request.prompt}"}