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
            template="""You are a professional financial analyst. Use the available tools to analyze stocks and return structured JSON.

Available tools: {tools}
Tool names: {tool_names}

IMPORTANT: Each tool has detailed documentation. Read the tool descriptions carefully to understand:
- What each tool does
- Required and optional parameters
- Expected output format

FORMAT RULES:
1. NO markdown (**, __) in Action/Action Input lines
2. Write ONE action, then STOP and wait for Observation
3. Use plain text only

CORRECT:
Action: search_stocks
Action Input: {{"query": "Apple"}}

WRONG:
**Action:** search_stocks
Action: **search_stocks**

WORKFLOW:
1. If ticker unknown → use search_stocks
2. Get stock data → use get_stock_info
3. For trends → use get_historical_prices (optional)
4. For news → use web_search (optional)
5. For analyst opinions → use get_analyst_recommendations (optional)

FINAL OUTPUT - Use "Final Answer:" followed by JSON:

Single stock:
Final Answer:
```json
{{
  "analysis_type": "single",
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "current_price": 178.25,
  "analysis": "Detailed analysis...",
  "metrics": {{"pe_ratio": 29.5, "market_cap": 2800000000000, "52week_high": 199.62, "52week_low": 164.08, "sector": "Technology", "industry": "Consumer Electronics"}},
  "period": "3mo",
  "analyst_recommendation": "Buy"
}}
```

Comparison (multiple stocks):
Final Answer:
```json
{{
  "analysis_type": "comparison",
  "stocks": [
    {{"ticker": "AAPL", "company_name": "Apple Inc.", "current_price": 178.25, "analysis": "...", "metrics": {{...}}, "analyst_recommendation": "Buy"}},
    {{"ticker": "MSFT", "company_name": "Microsoft Corporation", "current_price": 420.50, "analysis": "...", "metrics": {{...}}, "analyst_recommendation": "Hold"}}
  ],
  "comparison_summary": "Overall insights...",
  "period": "3mo"
}}
```

User Query: {input}

{agent_scratchpad}""",
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
        )

        # Report Generator 프롬프트
        self._prompts["report_generator"] = PromptTemplate(
            template="""You are a professional financial report writer. Generate comprehensive reports from analysis data using available tools when requested.

Available tools: {tools}
Tool names: {tool_names}

IMPORTANT: Each tool has detailed documentation explaining its purpose, parameters, and output.

FORMAT RULES:
1. NO markdown (**, __) in Action/Action Input lines
2. Write ONE action, then STOP and wait for Observation
3. Use plain text only

WORKFLOW:

STEP 1: Check user request for specific keywords
- Chart keywords: 차트, 그래프, chart, 그려, 시각화
- Save keywords: 저장, 파일, save, .md, .pdf, .txt

RULE: Only use tools if keywords are present!

STEP 2: Generate charts (ONLY if chart keywords found)
- Use draw_stock_chart for price charts
- Use draw_valuation_radar for valuation analysis (optional)

STEP 3: Write comprehensive report (ALWAYS)
Structure for single stock:
## [Company] ([TICKER]) 주식 분석 보고서
### 1. 기업 개요
### 2. 주가 정보
### 3. 밸류에이션 지표
### 4. 분석 의견
### 5. 투자 의견

Structure for comparison:
## 주식 비교 분석 보고서
### 1. 비교 대상
### 2. 주가 비교
### 3. 밸류에이션 비교
### 4. 종합 분석
### 5. 투자 추천

STEP 4: Save file (ONLY if save keywords found)
- Default format: md
- If ".pdf" mentioned → format="pdf"
- If ".txt" mentioned → format="txt"
- Include chart_paths if charts were generated

STEP 5: Final Answer
If NO save: Return report text
If saved: Return confirmation message

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
