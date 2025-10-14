"""Health check endpoint"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def healthcheck():
    """API health check"""
    return {"status": "healthy", "model": "veah-7b"}