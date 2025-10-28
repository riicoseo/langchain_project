class VectorStore:
    """텍스트 분할 및 벡터 DB 저장/검색"""
    def __init__(self, config):
        self.config = config
        self.embeddings = None
        self.vector_db = None  # ChromaDB
    
    def chunk_text(self, text, metadata):
        """텍스트 청킹"""
        pass
    
    def create_embeddings(self, chunks):
        """임베딩 생성"""
        pass
    
    def add_documents(self, documents):
        """문서를 벡터 DB에 저장"""
        pass
    
    def similarity_search(self, query, top_k=5):
        """유사도 기반 검색"""
        pass