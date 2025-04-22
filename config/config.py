from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database settings
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432

    # Model settings
    EMBEDDING_DIM: int = 3072
    AZURE_DEPLOYMENT: str = "text-embedding-3-large"
    OPENAI_API_VERSION: str = "2024-02-15-preview"
    AZURE_ENDPOINT: str = "https://YOUR_RESOURCE_NAME.openai.azure.com/"
    AZURE_API_KEY: str = "YOUR_AZURE_OPENAI_KEY"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    OPENAI_API_KEY: str

    # GPT-4 settings
    AZURE_CHAT_ENDPOINT: str = "gpt-4"
    AZURE_CHAT_DEPLOYMENT: str
    AZURE_CHAT_API_VERSION: str
    AZURE_CHAT_API_KEY: str

    # Search settings
    DEFAULT_BM25_WEIGHT: float = 0.4
    DEFAULT_CONTENT_VECTOR_WEIGHT: float = 0.4
    DEFAULT_TITLE_VECTOR_WEIGHT: float = 0.2
    DEFAULT_TOP_K: int = 10
    DEFAULT_RERANK: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
