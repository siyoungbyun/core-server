import logging
import nltk
from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi
from core.embedding import embedding_model
from models.rag import RAGDocument
from core.database import SessionLocal
import numpy as np
from .repository import RAGRepository
from langchain_openai import AzureChatOpenAI
from config.config import settings
from konlpy.tag import Okt

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.embedding_model = embedding_model
        self.db = SessionLocal()
        self.repository = RAGRepository()
        self.tokenizer = Okt()
        self.gpt4 = AzureChatOpenAI(
           azure_endpoint=settings.AZURE_CHAT_ENDPOINT,
           deployment_name=settings.AZURE_CHAT_DEPLOYMENT,
           openai_api_version=settings.AZURE_CHAT_API_VERSION,
           openai_api_key=settings.AZURE_CHAT_API_KEY,
        )
        
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 search using Korean tokenizer.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        try:
            # Extract nouns and verbs for better search
            tokens = self.tokenizer.nouns(text) + self.tokenizer.verbs(text)
            return tokens
        except Exception as e:
            logger.error(f"Error tokenizing Korean text: {str(e)}")
            # Fallback to simple space-based tokenization
            return text.split()
        
    async def chunk_text_context_aware(self, text: str, max_tokens: int = 4000, overlap: int = 200) -> List[str]:
        """
        Split text into chunks using GPT-4 to identify natural breakpoints while respecting token limits.
        
        Args:
            text (str): Text to chunk
            max_tokens (int): Maximum number of tokens per chunk
            overlap (int): Number of tokens to overlap between chunks
            
        Returns:
            List[str]: List of text chunks
        """
        # First, do a rough split to get initial chunks
        rough_chunks = self.chunk_text(text, max_tokens, overlap)
        
        # If text is small enough, return as is
        if len(rough_chunks) == 1:
            return rough_chunks
            
        refined_chunks = []
        
        for i, chunk in enumerate(rough_chunks):
            # Use GPT-4 to find the best breakpoint within the chunk
            prompt = f"""
            Given the following text, identify the best natural breakpoint that:
            1. Maintains semantic coherence
            2. Falls within the first {max_tokens} tokens
            3. Prefers breaking at paragraph boundaries or sentence endings
            
            Text:
            {chunk}
            
            Please respond with just the index of the character where the text should be split.
            If no good breakpoint is found, respond with -1.
            """
            
            try:
                response = await self.gpt4.ainvoke(prompt)
                breakpoint_idx = int(response.content.strip())
                
                if breakpoint_idx > 0 and breakpoint_idx < len(chunk):
                    # Split at the identified breakpoint
                    refined_chunks.append(chunk[:breakpoint_idx])
                    # Add the remaining text to the next chunk with overlap
                    if i < len(rough_chunks) - 1:
                        rough_chunks[i + 1] = chunk[breakpoint_idx - overlap:] + rough_chunks[i + 1]
                else:
                    refined_chunks.append(chunk)
                    
            except Exception as e:
                logger.error(f"Error in context-aware chunking: {str(e)}")
                # Fallback to original chunk if GPT-4 fails
                refined_chunks.append(chunk)
                
        return refined_chunks
        
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Maximum number of tokens per chunk
            overlap (int): Number of tokens to overlap between chunks
            
        Returns:
            List[str]: List of text chunks
        """
        tokens = self.tokenize_text(text)
        chunks = []
        
        if len(tokens) <= chunk_size:
            return [text]
            
        current_chunk = []
        current_size = 0
        
        for i, token in enumerate(tokens):
            current_chunk.append(token)
            current_size += 1
            
            if current_size >= chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Move back by overlap tokens
                current_chunk = current_chunk[-overlap:]
                current_size = len(current_chunk)
                
        # Add the last chunk if there are remaining tokens
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            
        return chunks
        
    async def process_document(self, title: str, content: str) -> RAGDocument:
        """
        Process a document by generating embeddings and storing in the database.
        
        Args:
            title (str): Document title
            content (str): Document content
            
        Returns:
            RAGDocument: Processed document with embeddings
        """
        try:
            # Generate embeddings for title and content
            title_embedding = await self.embedding_model.aembed_query(title)
            content_embedding = await self.embedding_model.aembed_query(content)
            
            # Tokenize content for BM25 search
            tokenized_text = self.tokenize_text(content)
            
            # Create document in database
            document = self.repository.create_document(
                title=title,
                content=content,
                tokenized_text=' '.join(tokenized_text),  # Join tokens with spaces
                title_embedding=title_embedding,
                content_embedding=content_embedding
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
        
    async def get_document(self, document_id: int) -> Optional[RAGDocument]:
        """
        Get a document by its ID.
        
        Args:
            document_id (int): Document ID
            
        Returns:
            Optional[RAGDocument]: Document if found, None otherwise
        """
        return self.repository.get_document_by_id(document_id)
        
    async def get_all_documents(self) -> List[RAGDocument]:
        """
        Get all documents.
        
        Returns:
            List[RAGDocument]: List of all documents
        """
        return self.repository.get_all_documents()
        
    async def delete_document(self, document_id: int) -> bool:
        """
        Delete a document by its ID.
        
        Args:
            document_id (int): Document ID
            
        Returns:
            bool: True if deleted, False if not found
        """
        return self.repository.delete_document(document_id)
        
    async def update_document(
        self,
        document_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None
    ) -> Optional[RAGDocument]:
        """
        Update a document's fields and regenerate embeddings if needed.
        
        Args:
            document_id (int): Document ID
            title (Optional[str]): New title
            content (Optional[str]): New content
            
        Returns:
            Optional[RAGDocument]: Updated document if found, None otherwise
        """
        try:
            title_embedding = None
            content_embedding = None
            
            # Generate new embeddings if title or content is updated
            if title is not None:
                title_embedding = await self.embedding_model.aembed_query(title)
            if content is not None:
                content_embedding = await self.embedding_model.aembed_query(content)
                
            return self.repository.update_document(
                document_id=document_id,
                title=title,
                content=content,
                title_embedding=title_embedding,
                content_embedding=content_embedding
            )
            
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            raise
        
    async def search_documents(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.4,
        content_vector_weight: float = 0.4,
        title_vector_weight: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Search documents using hybrid BM25 and vector search.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            bm25_weight (float): Weight for BM25 score
            content_vector_weight (float): Weight for content vector similarity
            title_vector_weight (float): Weight for title vector similarity
            
        Returns:
            List[Dict[str, Any]]: List of search results with scores
        """
        # Get all documents
        documents = self.db.query(RAGDocument).all()
        
        if not documents:
            return []
            
        # Tokenize query for BM25
        query_tokens = self.tokenize_text(query)
        
        # Prepare BM25
        tokenized_docs = [doc.tokenized_text.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        
        # Get BM25 scores
        bm25_scores = bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)
        
        # Get query embedding
        query_embedding = await self.embedding_model.aembed_query(query)
        
        # Calculate vector similarities
        content_similarities = []
        title_similarities = []
        
        for doc in documents:
            content_similarity = np.dot(query_embedding, doc.content_embedding)
            title_similarity = np.dot(query_embedding, doc.title_embedding)
            
            content_similarities.append(content_similarity)
            title_similarities.append(title_similarity)
            
        # Normalize similarities
        content_similarities = np.array(content_similarities)
        title_similarities = np.array(title_similarities)
        
        content_similarities = (content_similarities - content_similarities.min()) / (content_similarities.max() - content_similarities.min() + 1e-6)
        title_similarities = (title_similarities - title_similarities.min()) / (title_similarities.max() - title_similarities.min() + 1e-6)
        
        # Calculate combined scores
        combined_scores = (
            bm25_weight * bm25_scores +
            content_vector_weight * content_similarities +
            title_vector_weight * title_similarities
        )
        
        # Get top k results
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            doc = documents[idx]
            results.append({
                'id': doc.id,
                'title': doc.title,
                'content': doc.content,
                'score': float(combined_scores[idx]),
                'bm25_score': float(bm25_scores[idx]),
                'content_similarity': float(content_similarities[idx]),
                'title_similarity': float(title_similarities[idx])
            })
            
        return results

