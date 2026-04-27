from fastapi import APIRouter, HTTPException, status
import logging
from app.schemas import (
    EmbedRequest, EmbedResponse,
    BatchEmbedRequest, BatchEmbedResponse,
    HealthResponse
)
from app.models import model_instance
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/embed", response_model=EmbedResponse, status_code=status.HTTP_200_OK)
async def embed_text(request: EmbedRequest):
    """
    Gera embedding para um único texto
    """
    try:
        logger.info(f"Gerando embedding para texto: {request.text[:50]}...")
        
        vector = model_instance.encode(request.text)
        
        return EmbedResponse(
            vector=vector,
            dims=len(vector),
            model=settings.embeddings_model
        )
    except Exception as e:
        logger.error(f"Erro ao gerar embedding: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao gerar embedding: {str(e)}"
        )

@router.post("/embed/batch", response_model=BatchEmbedResponse, status_code=status.HTTP_200_OK)
async def embed_batch(request: BatchEmbedRequest):
    """
    Gera embeddings para múltiplos textos (batch)
    """
    try:
        # Valida tamanho do batch
        if len(request.texts) > settings.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size excede máximo de {settings.max_batch_size}"
            )
        
        logger.info(f"Gerando embeddings batch para {len(request.texts)} textos")
        
        vectors = model_instance.encode_batch(request.texts)
        
        return BatchEmbedResponse(
            vectors=vectors,
            dims=len(vectors[0]) if vectors else 0,
            model=settings.embeddings_model
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao gerar embeddings batch: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao gerar embeddings: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Verifica saúde do serviço
    """
    try:
        # Testa modelo com um texto simples
        test_vector = model_instance.encode("health check")
        
        return HealthResponse(
            status="ok",
            model=settings.embeddings_model,
            dims=len(test_vector)
        )
    except Exception as e:
        logger.error(f"Health check falhou: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Serviço indisponível"
        )