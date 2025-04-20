from fastapi import UploadFile
import uuid
from .repository import UploadRepository
from utils.video_processor import VideoProcessor
from .constants import ProcessingStatus
import os

class UploadService:
    def __init__(self):
        self.repository = UploadRepository()

    async def process_video_upload(
        self,
        file: UploadFile,
        title: str
    ):
        # 파일 저장 경로 생성
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        # 파일 저장
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 메타데이터 저장
        video = self.repository.save_video_metadata(
            title=title,
            file_path=file_path
        )
        
        # TODO: 비디오 처리 (백그라운드 작업으로 처리)
        # 현재는 간소화를 위해 생략
        
        return {"video_id": str(video.id)}

    def get_processing_status(self, video_id: int):
        status = self.repository.get_video_status(video_id)
        if not status:
            raise ValueError(f"Video with id {video_id} not found")
        return status 