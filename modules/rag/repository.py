from typing import List, Optional
from models.rag import RAGDocument
from core.database import SessionLocal
from konlpy.tag import Okt
import logging

logger = logging.getLogger(__name__)

class RAGRepository:
    def __init__(self):
        self.db = SessionLocal()
        self.tokenizer = Okt()
        
    def tokenize_korean_text(self, text: str) -> str:
        """
        Tokenize Korean text using Okt tokenizer.
        
        Args:
            text (str): Korean text to tokenize
            
        Returns:
            str: Tokenized text with spaces between tokens
        """
        try:
            # Extract nouns and verbs for better search
            tokens = self.tokenizer.nouns(text) + self.tokenizer.verbs(text)
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"Error tokenizing Korean text: {str(e)}")
            # Fallback to simple space-based tokenization
            return ' '.join(text.split())
            
    def create_document(
        self,
        title: str,
        content: str,
        title_embedding: List[float],
        content_embedding: List[float]
    ) -> RAGDocument:
        """
        Create a new RAG document with tokenized text.
        
        Args:
            title (str): Document title
            content (str): Document content
            title_embedding (List[float]): Title embedding vector
            content_embedding (List[float]): Content embedding vector
            
        Returns:
            RAGDocument: Created document
        """
        # Tokenize content for BM25
        tokenized_text = self.tokenize_korean_text(content)
        
        document = RAGDocument(
            title=title,
            content=content,
            tokenized_text=tokenized_text,
            title_embedding=title_embedding,
            content_embedding=content_embedding
        )
        
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        
        return document
        
    def get_document_by_id(self, document_id: int) -> Optional[RAGDocument]:
        """
        Get a document by its ID.
        
        Args:
            document_id (int): Document ID
            
        Returns:
            Optional[RAGDocument]: Document if found, None otherwise
        """
        return self.db.query(RAGDocument).filter(RAGDocument.id == document_id).first()
        
    def get_all_documents(self) -> List[RAGDocument]:
        """
        Get all documents.
        
        Returns:
            List[RAGDocument]: List of all documents
        """
        return self.db.query(RAGDocument).all()
        
    def delete_document(self, document_id: int) -> bool:
        """
        Delete a document by its ID.
        
        Args:
            document_id (int): Document ID
            
        Returns:
            bool: True if deleted, False if not found
        """
        document = self.get_document_by_id(document_id)
        if document:
            self.db.delete(document)
            self.db.commit()
            return True
        return False
        
    def update_document(
        self,
        document_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        title_embedding: Optional[List[float]] = None,
        content_embedding: Optional[List[float]] = None
    ) -> Optional[RAGDocument]:
        """
        Update a document's fields.
        
        Args:
            document_id (int): Document ID
            title (Optional[str]): New title
            content (Optional[str]): New content
            title_embedding (Optional[List[float]]): New title embedding
            content_embedding (Optional[List[float]]): New content embedding
            
        Returns:
            Optional[RAGDocument]: Updated document if found, None otherwise
        """
        document = self.get_document_by_id(document_id)
        if not document:
            return None
            
        if title is not None:
            document.title = title
        if content is not None:
            document.content = content
            document.tokenized_text = self.tokenize_korean_text(content)
        if title_embedding is not None:
            document.title_embedding = title_embedding
        if content_embedding is not None:
            document.content_embedding = content_embedding
            
        self.db.commit()
        self.db.refresh(document)
        
        return document 