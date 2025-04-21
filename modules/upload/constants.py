# 허용된 비디오 파일 형식
ALLOWED_VIDEO_TYPES = [
    'video/mp4',
    'video/x-msvideo',  # AVI
    'video/quicktime',  # MOV
    'video/x-ms-wmv',   # WMV
]

# 처리 상태 정의
class ProcessingStatus:
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"