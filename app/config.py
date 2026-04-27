from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    # Model configuration
    embeddings_model: str = "sentence-transformers/all-mpnet-base-v2"
    expected_dims: int = 768
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    timeout_seconds: int = 30
    
    # Batch configuration
    max_batch_size: int = 32
    prefer_batch: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Validação básica
if settings.expected_dims != 768:
    print(f"⚠️  Atenção: Dimensão esperada {settings.expected_dims} é diferente do padrão 768")