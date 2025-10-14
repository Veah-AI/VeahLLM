"""Explanation endpoints"""

from fastapi import APIRouter

router = APIRouter()

@router.post("/")
async def explain(code: str):
    """Explain Solana code"""
    return {"explanation": f"This code does..."}