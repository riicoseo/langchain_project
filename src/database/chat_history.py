class ChatHistory:
    """대화 기록 관리"""
    def __init__(self):
        self.history = []
        self.max_history = 5  # minimum
    
    def add_message(self, role, content):
        """메시지 추가"""
        pass
    
    def get_context(self):
        """대화 맥락 반환"""
        pass
    
    def summarize_history(self):
        """대화 기록 요약"""
        pass