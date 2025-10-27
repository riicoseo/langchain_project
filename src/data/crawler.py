class Crawler:
    """데이터 수집"""
    def __init__(self):
        self.client = None
        self.scraper = None
    
    def crawl_data(self, categories=[]):
        """데이터 수집"""
        pass
    
    def process_documents(self, documents):
        """문서 전처리"""
        pass

    def save_data(self, data, filepath):
        """수집 데이터 저장 (JSON)"""
        pass