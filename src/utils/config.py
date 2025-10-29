import os

class Config:
    """환경 변수 및 설정 관리"""
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SOLAR_API_KEY = os.getenv("SOLAR_API_KEY")
    
    # Vector DB
    PERSIST_DIR = "data/chroma_store"
    COLLECTION_NAME = "finance_terms"
    EMBEDDING_MODEL = "BAAI/bge-m3"
    PDF_PATH_PATTERN = "data/pdf/*.pdf"
    
    # Chunking
    CHUNK_SIZE_S = 300
    CHUNK_OVERLAP_S = 50
    CHUNK_SIZE_L = 800
    CHUNK_OVERLAP_L = 140
    
    # Retrieval
    TOP_K = 5
    


