from src.data.retriever import Retriever
from src.model.llm import LLM

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = LLM()
    
    def query(self, question: str) -> str:
        # 검색 → 생성 → 응답
        pass