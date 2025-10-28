class Retriever:
    """검색 및 재순위화"""
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def retrieve(self, query, top_k=5):
        """기본 검색"""
        return self.vector_store.similarity_search(query, top_k)
    
    def rerank(self, query, results):
        """재순위화"""
        pass