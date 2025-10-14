"""Solana RPC interface for VEAH"""

from solana.rpc.api import Client
from solana.publickey import PublicKey

class SolanaRPC:
    def __init__(self, endpoint="https://api.mainnet-beta.solana.com"):
        self.client = Client(endpoint)

    def get_transaction(self, signature: str):
        """Fetch transaction details"""
        return self.client.get_transaction(signature)

    def get_balance(self, address: str):
        """Get SOL balance"""
        pubkey = PublicKey(address)
        return self.client.get_balance(pubkey)

    def get_recent_transactions(self, address: str, limit=10):
        """Get recent transactions for address"""
        pubkey = PublicKey(address)
        return self.client.get_signatures_for_address(pubkey, limit=limit)