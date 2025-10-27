class Logger:
    """로깅 관리"""
    def __init__(self, log_file="app.log"):
        self.log_file = log_file
    
    def log_interaction(self, query, response, sources):
        """QA 상호작용 로깅"""
        pass
    
    def log_error(self, error):
        """에러 로깅"""
        pass