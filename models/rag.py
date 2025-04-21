from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Text, Index
from core.database import Base


class RAGDocument(Base):
    __tablename__ = "rag_documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255))
    content = Column(Text)
    tokenized_text = Column(Text)  # For BM25 search
    content_embedding = Column(Vector(1536))  # For vector search
    title_embedding = Column(Vector(1536))  # For title vector search

    # Index for BM25 search
    __table_args__ = (
        # GIN index for full-text search on tokenized_text
        Index('ix_documents_tokenized_text_gin', 'tokenized_text', postgresql_using='gin', postgresql_ops={'tokenized_text': 'gin_trgm_ops'}),
        # Index for vector similarity search on content_embedding
        Index('ix_documents_content_embedding', 'content_embedding', postgresql_using='ivfflat'),
        # Index for vector similarity search on title_embedding
        Index('ix_documents_title_embedding', 'title_embedding', postgresql_using='ivfflat'),
    )

    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title}')>" 