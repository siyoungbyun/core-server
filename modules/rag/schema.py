from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    rerank: Optional[bool] = False


class SearchResult(BaseModel):
    id: int
    title: str
    content: str
    score: float
    created_at: datetime


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int
    query_time: float


class AnswerResponse(BaseModel):
    answer: str
    query_time: float
    total_documents: int
