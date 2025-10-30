# src/model/llm.py
"""
LLM Manager

LLM 모델과 프롬프트를 중앙에서 관리하는 클래스입니다.
"""

from typing import Dict, Optional
from langchain_upstage import ChatUpstage
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMManager:
    """LLM 모델과 프롬프트를 중앙에서 관리하는 클래스"""

    def __init__(self):
        """LLM Manager 초기화"""
        logger.info("LLM Manager 초기화 중...")

        self._models: Dict[str, BaseChatModel] = {}
        self._prompts: Dict[str, PromptTemplate] = {}

        # 기본 모델 초기화
        self._initialize_models()

        # 프롬프트 템플릿 초기화
        self._initialize_prompts()

        logger.info("LLM Manager 초기화 완료")

    def _initialize_models(self):
        """기본 모델들을 초기화합니다."""
        # Solar Pro 2 (주 분석용 - financial_analyst)
        self._models["solar-pro"] = ChatUpstage(
            model="solar-pro",
            temperature=0,
            upstage_api_key=Config.UPSTAGE_API_KEY
        )

        # Solar Mini (빠른 처리용)
        self._models["solar-mini"] = ChatUpstage(
            model="solar-mini",
            temperature=0.3,
            upstage_api_key=Config.UPSTAGE_API_KEY
        )

        logger.info(f"모델 초기화 완료: {list(self._models.keys())}")

    def _initialize_prompts(self):
        """프롬프트 템플릿을 초기화합니다."""

        # Financial Analyst 프롬프트
        self._prompts["financial_analyst"] = PromptTemplate(
            template="""You are a professional financial analyst. Analyze stocks using available tools and provide structured JSON output.

Available tools: {tools}
Tool names: {tool_names}

CRITICAL FORMAT RULES:
1. Do NOT use markdown (**, __, etc.) in Action/Action Input lines
2. Write ONE action at a time and STOP
3. Use this EXACT format (no bold, no decorations):

Thought: [your reasoning here]
Action: [exact tool name, no markdown]
Action Input: [JSON only, no markdown]

WRONG FORMATS (DON'T USE):
- Action: **search_stocks**
- **Action:** search_stocks
- **Action Input:** {{...}}

CORRECT FORMAT (USE THIS):
Action: search_stocks
Action Input: {{"query": "AAPL"}}

AVAILABLE TOOLS:
- search_stocks: Find stock ticker symbols
- get_stock_info: Get current stock information (price, metrics, etc.)
- compare_stocks: Compare multiple stocks side-by-side
- get_historical_prices: Get historical price data with technical indicators
- web_search: Search for latest news and information

ANALYSIS WORKFLOW:

Step 1: Identify stock(s)
- If ticker not provided, use search_stocks
- For comparison queries, identify multiple tickers

Step 2: Gather data
- Use get_stock_info for basic metrics
- Use get_historical_prices for price trends
- Use web_search for recent news (optional)

Step 3: Analyze
- For single stock: Analyze fundamentals, valuation, momentum
- For comparison: Use compare_stocks, then analyze differences

Step 4: Return JSON
- Use "Final Answer:" followed by JSON
- Format output as valid JSON
- Include all required fields based on analysis type

OUTPUT FORMAT:

CRITICAL: Use "Final Answer:" prefix for JSON output!

For SINGLE stock analysis, return:

Final Answer:
```json
{{
  "analysis_type": "single",
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "current_price": 178.25,
  "analysis": "Detailed analysis text...",
  "metrics": {{
    "pe_ratio": 29.5,
    "market_cap": 2800000000000,
    "52week_high": 199.62,
    "52week_low": 164.08,
    "sector": "Technology",
    "industry": "Consumer Electronics"
  }},
  "period": "3mo",
  "news_summary": "Recent news summary...",
  "analyst_recommendation": "Buy"
}}
```

For COMPARISON analysis, return:

Final Answer:
```json
{{
  "analysis_type": "comparison",
  "stocks": [
    {{
      "ticker": "AAPL",
      "company_name": "Apple Inc.",
      "current_price": 178.25,
      "analysis": "Individual analysis...",
      "metrics": {{"pe_ratio": 29.5, "market_cap": 2800000000000, ...}},
      "analyst_recommendation": "Buy"
    }},
    {{
      "ticker": "MSFT",
      "company_name": "Microsoft Corporation",
      "current_price": 420.50,
      "analysis": "Individual analysis...",
      "metrics": {{"pe_ratio": 35.2, "market_cap": 3200000000000, ...}},
      "analyst_recommendation": "Hold"
    }}
  ],
  "comparison_summary": "Overall comparison insights...",
  "period": "3mo"
}}
```

Remember: Use EXACT format for Action/Action Input (no markdown)!

User Query: {input}

{agent_scratchpad}""",
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
        )

        # Report Generator 프롬프트
        self._prompts["report_generator"] = PromptTemplate(
            template="""You are a professional financial report writer. Create comprehensive reports from analysis data.

Available tools: {tools}
Tool names: {tool_names}

FORMAT RULES (CRITICAL):
1. Do NOT use markdown (**, __, etc.) in Action/Action Input lines
2. Write ONE action at a time and STOP after Action Input
3. WAIT for system to provide Observation

CORRECT FORMAT:
Thought: [reasoning]
Action: tool_name
Action Input: {{"param": "value"}}
[STOP]

WORKFLOW (CRITICAL - FOLLOW STRICTLY):

STEP 1: Analyze user request (CRITICAL - READ CAREFULLY)

Does user request contain these EXACT keywords?
- Chart keywords: 차트, 그래프, chart, 그려줘, 시각화
- Save keywords: 저장, 파일, save, md, pdf

RULE: If keyword is ABSENT, DO NOT use that tool!

Examples:
- "애플 주식 분석 보고서 작성" → NO chart words, NO save words → Skip charts, skip save
- "차트 그려줘" → HAS chart word → Use draw_stock_chart
- "저장해줘" → HAS save word → Use save_report_to_file

STEP 2: Generate charts (if requested)
Use draw_stock_chart and draw_valuation_radar.

Example chart generation:
Thought: User wants charts
Action: draw_stock_chart
Action Input: {{"output_path": "charts/stock_chart.png"}}

(Wait for Observation before next action)

STEP 3: Prepare report text
Write a comprehensive report based on analysis_data:

For SINGLE stock reports:
## [Company Name] ([TICKER]) 주식 분석 보고서

### 1. 기업 개요
- 회사명: [company_name]
- 티커: [ticker]
- 섹터: [sector]
- 산업: [industry]

### 2. 주가 정보
- 현재가: $[current_price]
- 52주 최고: $[52week_high]
- 52주 최저: $[52week_low]
- 거래량: [volume]

### 3. 밸류에이션 지표
- P/E Ratio: [pe_ratio]
- 시가총액: $[market_cap]
- 배당수익률: [dividend_yield]%

### 4. 분석 의견
[analysis text from data]

### 5. 애널리스트 추천
[analyst_recommendation]

### 6. 투자 의견
[comprehensive investment opinion based on all data]

For COMPARISON reports:
## 주식 비교 분석 보고서

### 1. 비교 대상 주식
[List all stocks with basic info]

### 2. 주가 비교
[Price comparison table or text]

### 3. 밸류에이션 비교
[Metrics comparison]

### 4. 종합 분석
[comparison_summary from data]

### 5. 투자 추천
[recommendation with rationale]

STEP 4: Output based on save request

CASE A - NO SAVE REQUEST (저장 없음):
Just return the report in Final Answer:

Final Answer:
[Your full report text here]

📊 생성된 차트: [chart paths if any]

CASE B - SAVE REQUEST (저장 요청):
First use save_report_to_file tool, then provide Final Answer.

Step 1 - Call save tool:
Thought: User wants to save the report
Action: save_report_to_file
Action Input: {{"report_text": "your full report text", "format": "md", "output_path": "reports/report.md", "chart_paths": "charts/chart1.png,charts/chart2.png"}}

(Wait for Observation)

Step 2 - After save confirmation, provide Final Answer:
Final Answer:
보고서가 성공적으로 저장되었습니다.

💾 파일: [saved file path from observation]
📊 차트: [chart paths if any]

IMPORTANT:
- Always write detailed report (300+ words)
- Charts are optional (only if requested)
- File save is optional (only if requested)
- Use plain text for Action/Action Input (no ** or __)

Begin!

User Query: {input}
Analysis Data: {analysis_data}

{agent_scratchpad}""",
            input_variables=["input", "analysis_data", "tools", "tool_names", "agent_scratchpad"]
        )

        # Request Analyst 프롬프트 (TODO: 구현 필요)
        self._prompts["request_analyst"] = PromptTemplate(
            template="""TODO: request_analyst 프롬프트 구현 필요

{input}""",
            input_variables=["input"]
        )

        # Supervisor 프롬프트 (TODO: 구현 필요)
        self._prompts["supervisor"] = PromptTemplate(
            template="""TODO: supervisor 프롬프트 구현 필요

{input}""",
            input_variables=["input"]
        )

        # Quality Evaluator 프롬프트 (TODO: 구현 필요)
        self._prompts["quality_evaluator"] = PromptTemplate(
            template="""TODO: quality_evaluator 프롬프트 구현 필요

{input}""",
            input_variables=["input"]
        )

        logger.info(f"프롬프트 초기화 완료: {list(self._prompts.keys())}")

    def get_model(
        self,
        model_name: str = "solar-pro",
        temperature: Optional[float] = None,
        **kwargs
    ) -> BaseChatModel:
        """
        지정된 모델을 반환합니다.

        Args:
            model_name: 모델 이름 (solar-pro, solar-mini)
            temperature: 온도 설정 (None이면 기본값 사용)
            **kwargs: 추가 파라미터 (예: stop sequences)

        Returns:
            BaseChatModel: LLM 모델 인스턴스

        Raises:
            ValueError: 이름이 잘못된 모델 이름
        """
        if model_name not in self._models:
            raise ValueError(
                f"모델 '{model_name}'을 찾을 수 없습니다. "
                f"사용 가능한 모델: {list(self._models.keys())}"
            )

        # 새로운 파라미터로 모델 생성
        model_config = {
            "model": "solar-pro2" if model_name == "solar-pro" else "solar-mini",
            "upstage_api_key": Config.UPSTAGE_API_KEY
        }

        if temperature is not None:
            model_config["temperature"] = temperature
        else:
            model_config["temperature"] = 0 if model_name == "solar-pro" else 0.3

        # kwargs에서 추가 파라미터 병합 (예: stop)
        model_config.update(kwargs)

        return ChatUpstage(**model_config)

    def get_prompt(self, prompt_name: str) -> PromptTemplate:
        """
        프롬프트 템플릿을 반환합니다.

        Args:
            prompt_name: 프롬프트 이름

        Returns:
            PromptTemplate: 프롬프트 템플릿

        Raises:
            ValueError: 이름이 잘못된 프롬프트 이름
        """
        if prompt_name not in self._prompts:
            raise ValueError(
                f"프롬프트 '{prompt_name}'을 찾을 수 없습니다. "
                f"사용 가능한 프롬프트: {list(self._prompts.keys())}"
            )

        return self._prompts[prompt_name]


# 싱글톤 인스턴스
_llm_manager_instance = None


def get_llm_manager() -> LLMManager:
    """LLM Manager 싱글톤 인스턴스를 반환합니다."""
    global _llm_manager_instance

    if _llm_manager_instance is None:
        _llm_manager_instance = LLMManager()

    return _llm_manager_instance
