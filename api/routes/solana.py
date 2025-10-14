"""Solana-specific endpoints"""

from fastapi import APIRouter

router = APIRouter()

@router.post("/transaction")
async def analyze_transaction(signature: str):
    """Analyze a Solana transaction"""
    return {"signature": signature, "analysis": "Transaction analysis here"}

@router.post("/wallet")
async def analyze_wallet(address: str):
    """Analyze a Solana wallet"""
    return {"address": address, "balance": "1000 SOL"}