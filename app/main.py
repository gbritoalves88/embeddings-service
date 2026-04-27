from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.router import router
from app.config import settings

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cria aplicação FastAPI
app = FastAPI(
    title="Serviço de Embeddings",
    description="Transforma texto em vetores (embeddings) para jurisprudência",
    version="1.0.0"
)

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restrinja para origens específicas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclui rotas
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """Executado quando a aplicação inicia"""
    logger.info("="*50)
    logger.info("Iniciando Serviço de Embeddings")
    logger.info(f"Modelo: {settings.embeddings_model}")
    logger.info(f"Porta: {settings.port}")
    logger.info(f"Dimensão esperada: {settings.expected_dims}")
    logger.info("="*50)

@app.on_event("shutdown")
async def shutdown_event():
    """Executado quando a aplicação para"""
    logger.info("Parando Serviço de Embeddings")

@app.get("/")
async def root():
    return {
        "service": "Embeddings Service",
        "version": "1.0.0",
        "status": "running",
        "model": settings.embeddings_model,
        "dims": settings.expected_dims
    }