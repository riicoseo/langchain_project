class LLMHandler:
    """LLM API 호출 및 응답 처리"""
    def __init__(self, config):
        self.config = config
        self.openai_client = None
        self.solar_client = None
    
    def call_llm(self, prompt, model="gpt-4o-mini"):
        """LLM API 호출 (OpenAI 또는 Solar)"""
        pass
    
    def format_prompt(self, query, context, chat_history):
        """프롬프트 템플릿 생성"""
        pass
    
    def parse_response(self, response):
        """응답 파싱 및 포맷팅"""
        pass