"""Solana-specific tokenizer for VEAH LLM"""

from transformers import AutoTokenizer
from typing import List, Dict

class SolanaTokenizer:
    def __init__(self, base_model="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self._add_solana_tokens()

    def _add_solana_tokens(self):
        """Add Solana-specific tokens to vocabulary"""
        solana_tokens = [
            # Currency
            "<SOL>", "<LAMPORT>", "<USDC>", "<USDT>", "<WSOL>",
            # Technical
            "<SLOT>", "<EPOCH>", "<BLOCKHASH>", "<PUBKEY>", "<SIGNATURE>",
            # Programs
            "<SYSTEM_PROGRAM>", "<TOKEN_PROGRAM>", "<METAPLEX>",
            "<RAYDIUM>", "<ORCA>", "<JUPITER>", "<MARINADE>",
            # Instructions
            "<TRANSFER>", "<SWAP>", "<STAKE>", "<MINT>", "<BURN>",
            # Special
            "<TX_START>", "<TX_END>", "<COMPUTE_UNITS>", "<PRIORITY_FEE>",
        ]
        self.tokenizer.add_tokens(solana_tokens)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)