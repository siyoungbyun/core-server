from fastapi import BackgroundTasks
from utils.stt_processor import process_video_to_text
from models.rag_document import RAGDocument
from openai import AzureOpenAI
from konlpy.tag import Okt
import logging
from core.database import SessionLocal
from config.config import settings
from .repository import UploadRepository

logger = logging.getLogger(__name__)

# Initialize Azure OpenAI client
openai_client = AzureOpenAI(
    api_key=settings.AZURE_API_KEY,
    api_version=settings.OPENAI_API_VERSION,
    azure_endpoint=settings.AZURE_ENDPOINT,
)

# Initialize Okt tokenizer
okt = Okt()


def preprocess_korean_text(text):
    """한국어 텍스트 전처리 및 토큰화 함수"""
    # Okt로 형태소 분석 및 토큰화
    tokens = []
    for word, pos in okt.pos(text):
        # 명사, 동사, 형용사, 부사만 추출
        if pos in ["Noun", "Verb", "Adjective", "Adverb"]:
            tokens.append(word)

    # 한 글자 토큰 제거
    tokens = [token for token in tokens if len(token) > 1]
    return " ".join(tokens)


async def process_embeddings(video_id: int, title: str, content: str):
    """텍스트를 벡터 임베딩으로 변환하고 토큰화하는 백그라운드 작업"""
    try:
        # 텍스트 전처리 및 토큰화
        tokenized_text = preprocess_korean_text(content)

        # Azure OpenAI 임베딩 생성
        try:
            content_embedding = (
                openai_client.embeddings.create(
                    model=settings.AZURE_DEPLOYMENT, input=content
                )
                .data[0]
                .embedding
            )

            title_embedding = (
                openai_client.embeddings.create(
                    model=settings.AZURE_DEPLOYMENT, input=title
                )
                .data[0]
                .embedding
            )
        except Exception as e:
            logger.error(f"Failed to create embeddings: {str(e)}")
            raise

        # 데이터베이스에 저장
        db = SessionLocal()
        try:
            rag_document = RAGDocument(
                title=title,
                content=content,
                tokenized_text=tokenized_text,
                content_embedding=content_embedding,
                title_embedding=title_embedding,
            )
            db.add(rag_document)
            db.commit()
            db.refresh(rag_document)
        finally:
            db.close()

        logger.info(f"[Embedding] 임베딩 처리 완료 (ID: {video_id})")
        return True

    except Exception as e:
        error_msg = f"[Embedding] 임베딩 처리 실패 (ID: {video_id}): {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False


async def process_video_transcript(video_id: int, file_path: str):
    """비디오 파일을 처리하여 텍스트로 변환하는 백그라운드 작업"""
    try:
        # STT 처리
        logger.info(f"[STT] 오디오 추출 및 변환 처리 시작...")
        transcription_result = process_video_to_text(file_path)

        # 전체 텍스트 추출
        full_text = transcription_result["text"]
        logger.info(f"[STT] 텍스트 변환 완료: {len(full_text)} 문자")

        # 데이터베이스에 저장
        repository = UploadRepository()
        repository.update_video_transcript(video_id, full_text)

        # 비디오 정보 가져오기
        video = repository.get_video_by_id(video_id)

        # 임베딩 처리
        await process_embeddings(
            video_id=video_id, title=video.title, content=full_text
        )

        logger.info(f"[STT] 비디오 처리 완료 (ID: {video_id})")
        return True

    except Exception as e:
        error_msg = f"[STT] 비디오 처리 실패 (ID: {video_id}): {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False
