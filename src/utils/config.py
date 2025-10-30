# src/utils/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    """환경 변수 및 설정 관리"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SOLAR_API_KEY = os.getenv("SOLAR_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # API Key 검증
    @classmethod
    def validate_api_keys(cls):
        """필수 API 키 존재 여부 확인"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        if not cls.SOLAR_API_KEY:
            raise ValueError("SOLAR_API_KEY가 설정되지 않았습니다.")
        if not cls.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY가 설정되지 않았습니다.")
            
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # Logs
    LOGS_DIR = BASE_DIR / "logs"
    
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
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    


