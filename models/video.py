from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from core.database import Base

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    file_path = Column(String(255), nullable=False)
    transcript = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "file_path": self.file_path,
            "transcript": self.transcript,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        } 