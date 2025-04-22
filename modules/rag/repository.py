from typing import List, Tuple
from sqlalchemy import func, text
from sqlalchemy.orm import Session
from models.rag_document import RAGDocument
from config.config import settings
from openai import AzureOpenAI
import time
import logging

logger = logging.getLogger(__name__)


class RAGRepository:
    def __init__(self, db: Session):
        self.db = db
        self.openai_client = AzureOpenAI(
            api_key=settings.AZURE_API_KEY,
            api_version=settings.OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_ENDPOINT,
        )

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for the query using Azure OpenAI's API"""
        response = self.openai_client.embeddings.create(
            model=settings.AZURE_DEPLOYMENT, input=query
        )
        return response.data[0].embedding

    def hybrid_search(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[RAGDocument, float]]:
        """Perform hybrid search combining BM25 and vector similarity"""
        start_time = time.time()

        # Get query embedding
        query_embedding = self._get_query_embedding(query)
        logger.info(f"Query embedding length: {len(query_embedding)}")

        # Check if we have any documents in the database
        doc_count = self.db.query(RAGDocument).count()
        logger.info(f"Total documents in database: {doc_count}")

        # First, get all documents with their raw similarity scores
        raw_results = (
            self.db.query(
                RAGDocument,
                (
                    1
                    - func.cosine_distance(
                        RAGDocument.content_embedding,
                        text("CAST(:query_embedding AS vector)"),
                    )
                ).label("content_similarity"),
                (
                    1
                    - func.cosine_distance(
                        RAGDocument.title_embedding,
                        text("CAST(:query_embedding AS vector)"),
                    )
                ).label("title_similarity"),
                func.ts_rank_cd(
                    text("to_tsvector('simple', tokenized_text)"),
                    text("plainto_tsquery('simple', :query)"),
                    32,  # normalization option
                ).label("bm25_score"),
            )
            .params(query=query, query_embedding=query_embedding)
            .all()
        )

        if not raw_results:
            return [], 0

        # Get min and max values for normalization
        content_similarities = [r[1] for r in raw_results]
        title_similarities = [r[2] for r in raw_results]
        bm25_scores = [r[3] for r in raw_results]

        content_min = min(content_similarities)
        content_max = max(content_similarities)
        title_min = min(title_similarities)
        title_max = max(title_similarities)
        bm25_min = min(bm25_scores)
        bm25_max = max(bm25_scores)

        # Calculate normalized and weighted scores
        results = []
        for doc, content_sim, title_sim, bm25_score in raw_results:
            # Normalize scores
            norm_content = (
                (content_sim - content_min) / (content_max - content_min)
                if content_max != content_min
                else 0
            )
            norm_title = (
                (title_sim - title_min) / (title_max - title_min)
                if title_max != title_min
                else 0
            )
            norm_bm25 = (
                (bm25_score - bm25_min) / (bm25_max - bm25_min)
                if bm25_max != bm25_min
                else 0
            )

            # Apply weights
            weighted_score = (
                settings.DEFAULT_CONTENT_VECTOR_WEIGHT * norm_content
                + settings.DEFAULT_TITLE_VECTOR_WEIGHT * norm_title
                + settings.DEFAULT_BM25_WEIGHT * norm_bm25
            )
            results.append((doc, weighted_score))

        # Sort by weighted score and get top_k
        results.sort(key=lambda x: x[1], reverse=True)
        final_results = results[:top_k]

        logger.info(f"Found {len(final_results)} results")
        query_time = time.time() - start_time
        return final_results, query_time
