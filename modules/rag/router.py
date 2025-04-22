from fastapi import APIRouter, Depends
from .service import RAGService
from .schema import SearchRequest, AnswerResponse

router = APIRouter(prefix="/rag", tags=["rag"])


# Dependency injection
def get_rag_service():
    return RAGService()


@router.post(
    "/query",
    response_model=AnswerResponse,
    summary="RAG 기반 질의응답",
    description="""
    RAG(Retrieval-Augmented Generation) 기반으로 질문에 대한 답변을 생성합니다.

    ## 동작 방식
    1. BM25 (0.4) + 내용 벡터 유사도 (0.4) + 제목 벡터 유사도 (0.2)의 가중치로 관련 문서 검색
    2. 검색된 문서를 컨텍스트로 활용하여 LLM이 답변 생성
    3. 한국어로 답변 제공

    ## 파라미터
    * query: 질문
    * top_k: 검색에 사용할 문서 수 (기본값: 5)
    """,
)
async def query(
    request: SearchRequest, service: RAGService = Depends(get_rag_service)
) -> AnswerResponse:
    """Generate answer using RAG"""
    return await service.generate_answer(query=request.query, top_k=request.top_k)
