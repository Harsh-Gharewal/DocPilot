"""FastAPI application entrypoint for DocPilot RAG System."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings

logger = logging.getLogger("docpilot_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    logger.info("Starting DocPilot RAG API Service...")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"Embedding Provider: {settings.embedding_provider}")
    yield
    logger.info("Shutting down DocPilot RAG API Service...")


app = FastAPI(
    title="DocPilot Enterprise RAG API",
    description="Production-grade hybrid search, verification, and grounded generation over internal documents.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend dashboard and developer tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/", tags=["Root"])
def root_endpoint():
    """Welcome root endpoint."""
    return {
        "service": "DocPilot RAG Engine",
        "docs_url": "/docs",
        "health_check": "/v1/health",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)

