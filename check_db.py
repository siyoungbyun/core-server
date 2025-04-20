from core.database import SessionLocal
from models.video import Video
from pprint import pprint

def check_videos_in_db():
    """데이터베이스에 저장된 비디오 목록을 확인합니다."""
    db = SessionLocal()
    videos = db.query(Video).all()
    
    if not videos:
        print("데이터베이스에 비디오가 없습니다.")
        return
    
    print(f"총 {len(videos)}개의 비디오가 저장되어 있습니다.")
    for video in videos:
        print("\n-----------------------------------")
        pprint(video.to_dict())
    
    db.close()

if __name__ == "__main__":
    check_videos_in_db() 