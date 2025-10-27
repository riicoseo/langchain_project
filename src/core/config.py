import os

class Config:
    """환경 변수 및 설정 관리"""
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SOLAR_API_KEY = os.getenv("SOLAR_API_KEY")
    
    # Vector DB
    VECTOR_DB_PATH = "./chroma_db"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    
    # Chunking
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval
    TOP_K = 5