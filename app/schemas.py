from pydantic import BaseModel, Field, validator
from typing import List, Optional

class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texto para gerar embedding")

    @validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Texto não pode ser vazio')
        return v.strip()

class EmbedResponse(BaseModel):
    vector: List[float]
    dims: int
    model: str

class BatchEmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=32, description="Lista de textos")

    @validator('texts')
    def texts_not_empty(cls, v):
        if not v:
            raise ValueError('Lista de textos não pode ser vazia')
        for text in v:
            if not text or not text.strip():
                raise ValueError('Texto não pode ser vazio')
        return [t.strip() for t in v]

class BatchEmbedResponse(BaseModel):
    vectors: List[List[float]]
    dims: int
    model: str

class HealthResponse(BaseModel):
    status: str
    model: str
    dims: int