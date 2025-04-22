from typing import List
from .repository import RAGRepository
from .schema import SearchResponse, SearchResult, AnswerResponse
from core.database import SessionLocal
from datetime import datetime
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from config.config import settings


class RAGService:
    def __init__(self):
        self.db = SessionLocal()
        self.repository = RAGRepository(self.db)
        self.llm = AzureChatOpenAI(
            temperature=0,
            deployment_name=settings.AZURE_CHAT_DEPLOYMENT,
            openai_api_version=settings.AZURE_CHAT_API_VERSION,
            openai_api_key=settings.AZURE_CHAT_API_KEY,
            azure_endpoint=settings.AZURE_ENDPOINT,
        )
        self.prompt = ChatPromptTemplate.from_template(
            """다음은 검색된 문서 내용입니다:

            {context}

            다음 질문에 대해 문서 내용을 바탕으로 답변해주세요:
            {question}

            답변은 한국어로 작성해주세요. 문서 내용을 바탕으로 구체적이고 명확하게 답변해주세요.
            """
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def search(self, query: str, top_k: int = 10) -> SearchResponse:
        """Perform hybrid search and return formatted results"""
        results, query_time = self.repository.hybrid_search(query, top_k)

        search_results = [
            SearchResult(
                id=doc.id,
                title=doc.title,
                content=doc.content,
                score=score,
                created_at=getattr(doc, "created_at", datetime.now()),
            )
            for doc, score in results
        ]

        return SearchResponse(
            results=search_results,
            total_results=len(search_results),
            query_time=query_time,
        )

    async def generate_answer(self, query: str, top_k: int = 5) -> AnswerResponse:
        """Generate an answer using LLM based on retrieved documents"""
        # Retrieve relevant documents
        results, query_time = self.repository.hybrid_search(query, top_k)

        # Prepare context from retrieved documents
        context = "\n\n".join(
            [f"문서 제목: {doc.title}\n내용: {doc.content}" for doc, _ in results]
        )

        # Generate answer using LLM
        answer = await self.chain.ainvoke({"context": context, "question": query})

        return AnswerResponse(
            answer=answer, query_time=query_time, total_documents=len(results)
        )
