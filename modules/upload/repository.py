from typing import Optional
from models.video import Video
from .schema import VideoProcessingStatus
from .constants import ProcessingStatus
from core.database import SessionLocal
from sqlalchemy.orm import Session

class UploadRepository:
    def __init__(self):
        self.db = SessionLocal()

    def save_video_metadata(
        self,
        title: str,
        file_path: str,
        status: str = None
    ) -> Video:
        video = Video(
            title=title,
            file_path=file_path,
            transcript=None
        )
        self.db.add(video)
        self.db.commit()
        self.db.refresh(video)
        return video

    def get_video_status(self, video_id: int) -> Optional[VideoProcessingStatus]:
        video = self.db.query(Video).filter(Video.id == video_id).first()
        if not video:
            return None
        
        # Assuming we don't have a status field in Video model anymore
        # We'll determine status based on transcript
        status = ProcessingStatus.COMPLETED if video.transcript else ProcessingStatus.PROCESSING
        
        return VideoProcessingStatus(
            video_id=str(video.id),
            status=status,
            progress=100 if status == ProcessingStatus.COMPLETED else 0,
            created_at=video.created_at,
            updated_at=video.updated_at
        )

    def create_video(self, title: str, file_path: str) -> Video:
        video = Video(
            title=title,
            file_path=file_path
        )
        self.db.add(video)
        self.db.commit()
        self.db.refresh(video)
        return video

    def get_video_by_id(self, video_id: int) -> Video:
        return self.db.query(Video).filter(Video.id == video_id).first()

    def update_video_transcript(self, video_id: int, transcript: str) -> Video:
        video = self.get_video_by_id(video_id)
        if video:
            video.transcript = transcript
            self.db.commit()
            self.db.refresh(video)
        return video 