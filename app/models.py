import os
import logging
from sentence_transformers import SentenceTransformer
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self):
        self.model_name = settings.embeddings_model
        self.expected_dims = settings.expected_dims
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo de embeddings (local ou HuggingFace)"""
        try:
            logger.info(f"Carregando modelo: {self.model_name}")
            
            # Verifica se é um caminho local
            if os.path.exists(self.model_name):
                logger.info(f"Carregando modelo do caminho local: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
            else:
                logger.info(f"Baixando modelo do HuggingFace: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
            
            # Testa com um texto simples para validar dimensão
            test_embedding = self.model.encode("teste")
            actual_dims = len(test_embedding)
            
            if actual_dims != self.expected_dims:
                logger.warning(f"Dimensão do modelo ({actual_dims}) diferente da esperada ({self.expected_dims})")
            else:
                logger.info(f"Modelo carregado com sucesso! Dimensão: {actual_dims}")
                
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise
    
    def encode(self, text: str):
        """Gera embedding para um único texto"""
        if not self.model:
            raise RuntimeError("Modelo não carregado")
        
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def encode_batch(self, texts: list):
        """Gera embeddings para múltiplos textos (batch)"""
        if not self.model:
            raise RuntimeError("Modelo não carregado")
        
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]

# Instância global do modelo
model_instance = EmbeddingModel()