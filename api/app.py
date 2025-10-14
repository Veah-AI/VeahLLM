"""VEAH LLM API Server"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import query, solana, explain, healthcheck

app = FastAPI(title="VEAH LLM API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(solana.router, prefix="/solana", tags=["Solana"])
app.include_router(explain.router, prefix="/explain", tags=["Explain"])
app.include_router(healthcheck.router, prefix="/health", tags=["Health"])

@app.get("/")
def root():
    return {
        "name": "VEAH LLM",
        "description": "Solana-Native Language Model",
        "version": "1.0.0"
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()