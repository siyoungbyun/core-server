from langchain_openai import AzureOpenAIEmbeddings
from config.config import settings

embedding_model = AzureOpenAIEmbeddings(
    azure_deployment=settings.AZURE_DEPLOYMENT,
    openai_api_version=settings.OPENAI_API_VERSION,
    azure_endpoint=settings.AZURE_ENDPOINT,
    api_key=settings.API_KEY,
)
