"""
Report Generator Agent

financial_analyst 또는 vector_search_agent의 출력을 받아서
보고서를 생성하고, 필요시 차트를 그리고, 파일로 저장하는 에이전트입니다.
"""

import json
import os

from typing import Dict, Any, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agents.tools.report_tools import report_tools
from src.model.llm import get_llm_manager
from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)

# Global variable to store current analysis_data for tools
_current_analysis_data_json = None


def _set_current_analysis_data(data_json: str):
    """Set the current analysis data for tools to access"""
    global _current_analysis_data_json
    _current_analysis_data_json = data_json


def _get_current_analysis_data() -> str:
    """Get the current analysis data JSON string"""
    global _current_analysis_data_json
    return _current_analysis_data_json if _current_analysis_data_json else "{}"


class ReportGenerator:
    def __init__(self, model_name: str = None, temperature: float = 0.0):
        """
        Report Generator 에이전트를 초기화합니다.

        Args:
            model_name: 사용할 모델명 (default: Config.LLM_MODEL)
            temperature: LLM 온도 (0.0 = 결정적)
        """
        if model_name is None:
            model_name = Config.LLM_MODEL
        logger.info(f"Report Generator 초기화 - model: {model_name}, temp: {temperature}")

        # LLM Manager에서 모델 가져오기 (stop sequence 포함)
        llm_manager = get_llm_manager()
        self.llm = llm_manager.get_model(
            model_name,
            temperature=temperature,
            stop=["\nObservation:", "Observation:"]  # Action 후에 멈추도록 강제
        )

        self.tools = report_tools
        self.agent_executor = self._create_agent()

        logger.info("Report Generator 초기화 완료")

    def _create_agent(self) -> AgentExecutor:
        """ReAct 에이전트를 생성합니다."""

        # LLM Manager에서 프롬프트 가져오기
        llm_manager = get_llm_manager()
        base_prompt = llm_manager.get_prompt("report_generator")
        prompt = ChatPromptTemplate.from_messages([
             ("system", base_prompt.template),
            ("system", "{agent_scratchpad}"),     # 문자열 슬롯로 변경
            ("human", "User Query: {input}")
        ])
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        def handle_parsing_error(error: Exception) -> str:
            error_msg = str(error)
            logger.warning(f"[PARSING ERROR] {error_msg[:150]}...")

            # 마크다운 관련 에러 감지
            if "** " in error_msg or " **" in error_msg:
                return """ERROR: You used markdown (** or __) in Action line!

WRONG:
**Action:** tool_name
Action: **tool_name**

CORRECT:
Action: tool_name
Action Input: {"key": "value"}

Remove ALL markdown from Action/Action Input lines!"""

            if "both a final answer and a parse-able action" in error_msg:
                return """ERROR: You wrote Action and Final Answer together!

Write ONE thing at a time:
1. Action: tool
2. Action Input: {...}
3. STOP and wait
4. (system provides Observation)
5. Continue or Final Answer"""

            if "not a valid tool" in error_msg:
                return """ERROR: Tool name has extra characters!

Valid tools EXACTLY:
- draw_stock_chart
- draw_valuation_radar
- save_report_to_file

Check for spaces, markdown, or typos!"""

            return "Format error. Use plain text for Action lines."
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,  # 차트 2개 + 저장 + 오류 여유 = 10
            early_stopping_method="force",  # generate → force로 변경
            handle_parsing_errors=handle_parsing_error,
            return_intermediate_steps=True
        )
        
        return agent_executor


    def generate_report(
        self,
        user_request: str,
        analysis_data: Dict[str, Any],
        messages:list = None,
    ) -> Dict[str, Any]:
        """
        분석 데이터를 기반으로 보고서를 생성합니다.
        """
        if messages == None:
            messages = []
        try:
            logger.info(f"보고서 생성 시작 - request: {user_request[:50]}...")

            if not analysis_data:
                logger.error("Analysis data is empty")
                return {
                    "report": "❌ 분석 데이터가 없습니다.",
                    "status": "error",
                    "charts": [],
                    "saved_path": None
                }

            # 요청 분석: 차트 및 저장 요청 감지
            request_lower = user_request.lower()

            chart_keywords = ['차트', '그래프', 'chart', '그려', '시각화']
            save_keywords = ['저장', '파일', 'save', 'md', 'pdf', 'txt']

            wants_charts = any(keyword in request_lower for keyword in chart_keywords)
            wants_save = any(keyword in request_lower for keyword in save_keywords)

            logger.info(f"요청 분석 - 차트: {wants_charts}, 저장: {wants_save}")

            # analysis_data를 JSON 문자열로 변환하여 프롬프트에 전달
            analysis_json = json.dumps(analysis_data, ensure_ascii=False, indent=2)

            # IMPORTANT: chart tools를 위해 분석 데이터를 JSON 문자열로 준비
            # LLM이 이 문자열을 그대로 도구에 전달해야 함
            # 임시로 글로벌 변수에 저장하여 도구가 접근할 수 있게 함
            _set_current_analysis_data(analysis_json)

            # 도구 필터링: 요청에 따라 사용 가능한 도구 제한
            available_tools = []
            if wants_charts:
                # 차트 요청 시에만 차트 도구 추가
                available_tools.extend([
                    tool for tool in self.tools
                    if tool.name in ['draw_stock_chart', 'draw_valuation_radar']
                ])
            if wants_save:
                # 저장 요청 시에만 저장 도구 추가
                available_tools.extend([
                    tool for tool in self.tools
                    if tool.name == 'save_report_to_file'
                ])

            logger.info(f"사용 가능한 도구: {[t.name for t in available_tools]}")

            # 도구가 없으면 에이전트를 사용하지 않고 직접 보고서 생성
            if not available_tools:
                logger.info("도구 없음 - 직접 보고서 생성")
                report_text = self._generate_report_directly(analysis_data)
                return {
                    "report": report_text,
                    "status": "success",
                    "charts": [],
                    "saved_path": None
                }

            # 필터링된 도구로 임시 에이전트 생성
            from langchain.agents import create_react_agent, AgentExecutor
            llm_manager = get_llm_manager()
            base_prompt = llm_manager.get_prompt("report_generator")

            prompt = ChatPromptTemplate.from_messages([
             ("system", base_prompt.template),
            ("system", "{agent_scratchpad}"),     # 문자열 슬롯로 변경
            ("human", "User Query: {input}")
        ])

            temp_agent = create_react_agent(
                llm=self.llm,
                tools=available_tools,
                prompt=prompt
            )

            temp_executor = AgentExecutor(
                agent=temp_agent,
                tools=available_tools,
                verbose=True,
                max_iterations=10,  # 차트 2개 + 저장 + 오류 여유 = 10
                early_stopping_method="force",
                handle_parsing_errors=self.agent_executor.handle_parsing_errors,
                return_intermediate_steps=True
            )

            result = temp_executor.invoke({
                "input": user_request,
                "analysis_data": analysis_json,
                "messages": messages
            })

            output = result.get("output", "보고서 생성에 실패했습니다.")
            intermediate_steps = result.get("intermediate_steps", [])

            # 개선된 차트/파일 경로 추출
            charts = []
            saved_path = None

            for action, observation in intermediate_steps:
                tool_name = action.tool
                obs_str = str(observation)

                logger.debug(f"Tool: {tool_name}, Observation: {obs_str[:100]}")

                if tool_name in ["draw_stock_chart", "draw_valuation_radar"]:
                    # "✓ 차트가 charts/xxx.png에 저장되었습니다" 형식
                    # OR "성공적으로 생성되었습니다: charts/xxx.png" 형식
                    import re

                    # 패턴 1: "charts/xxx.png에"
                    match = re.search(r'(charts/[^\s]+\.png)', obs_str)
                    if match:
                        chart_path = match.group(1)
                        if chart_path not in charts:
                            charts.append(chart_path)
                            logger.info(f"차트 경로 추출: {chart_path}")

                    # 패턴 2: ": charts/xxx.png"
                    match2 = re.search(r':\s*(charts/[^\s]+\.png)', obs_str)
                    if match2 and match2.group(1) not in charts:
                        charts.append(match2.group(1))

                elif tool_name == "save_report_to_file":
                    # "✓ 보고서가 reports/xxx.md에 저장"
                    # OR "reports/xxx.pdf에 저장되었습니다"
                    import re
                    match = re.search(r'(reports/[^\s]+\.(txt|md|pdf))', obs_str)
                    if match:
                        saved_path = match.group(1)
                        logger.info(f"저장 경로 추출: {saved_path}")

            logger.info(f"보고서 생성 완료 - charts: {len(charts)}, saved: {saved_path is not None}")

            return {
                "report": output,
                "status": "success",
                "charts": charts,
                "saved_path": saved_path
            }

        except Exception as e:
            logger.error(f"보고서 생성 실패: {str(e)}")
            import traceback
            logger.debug(f"상세 에러:\n{traceback.format_exc()}")

            error_report = f"""# Report Generation Error

An error occurred: {str(e)}

## Analysis Data
```json
{json.dumps(analysis_data, ensure_ascii=False, indent=2)}
```

⚠️ Supervisor에게 재시도를 요청하거나 데이터를 확인해주세요."""

            return {
                "report": error_report,
                "status": "error",
                "charts": [],
                "saved_path": None,
                "error": str(e)
            }

    def _generate_report_directly(self, analysis_data: Dict[str, Any]) -> str:
        """
        에이전트 없이 직접 보고서를 생성합니다 (도구 불필요한 경우).

        Args:
            analysis_data: 분석 데이터 딕셔너리

        Returns:
            str: 마크다운 형식의 보고서 텍스트
        """
        logger.info("직접 보고서 생성 시작")

        analysis_type = analysis_data.get("analysis_type", "single")
        llm_manager = get_llm_manager()

        try:
            if analysis_type == "single":
                # 단일 주식 보고서 생성
                analysis_json = json.dumps(analysis_data, ensure_ascii=False, indent=2)
                prompt_template = llm_manager.get_prompt("report_direct_single")
                prompt = prompt_template.format(analysis_json=analysis_json)

            elif analysis_type == "comparison":
                # 비교 분석 보고서 생성
                stocks = analysis_data.get("stocks", [])
                tickers = [s.get("ticker") for s in stocks]
                tickers_str = " vs ".join(tickers)

                analysis_json = json.dumps(analysis_data, ensure_ascii=False, indent=2)
                prompt_template = llm_manager.get_prompt("report_direct_comparison")
                prompt = prompt_template.format(
                    analysis_json=analysis_json,
                    tickers=tickers_str
                )

            elif analysis_type == "rag":
                documents = analysis_data.get("documents") or []
                query = analysis_data.get("query", "")

                if not documents:
                    logger.warning("RAG 검색 결과가 비어 있습니다.")
                    return (
                        "## RAG 검색 결과 없음\n\n"
                        f"질문: {query or '질문 정보 없음'}\n\n"
                        "검색된 문서가 없어 보고서를 생성할 수 없습니다. "
                        "질문을 더 구체적으로 작성하거나 다른 키워드로 다시 시도해 주세요."
                    )

                max_docs = 1
                selected_docs = documents[:max_docs]
                if len(documents) > max_docs:
                    logger.info(
                        "RAG 문서 %d개 중 상위 %d개만 사용합니다.",
                        len(documents),
                        max_docs
                    )

                documents_block = "\n\n".join(
                    f"[문서 {idx + 1}]\n{doc}"
                    for idx, doc in enumerate(selected_docs)
                )

                prompt_template = llm_manager.get_prompt("report_direct_rag")
                prompt = prompt_template.format(
                    query=query,
                    documents_block=documents_block
                )

            elif analysis_type in ["general", "definition", "explanation", "concept"]:
                # financial_analyst가 일반 금융 용어/개념 설명 반환 시
                query = analysis_data.get("query", "")

                # 여러 가능한 필드에서 내용 찾기
                analysis_text = (
                    analysis_data.get("analysis") or
                    analysis_data.get("explanation") or
                    analysis_data.get("description") or
                    analysis_data.get("answer") or
                    ""
                )

                # 필드가 비어있으면 analysis_data 전체를 사용
                if not analysis_text:
                    logger.warning(f"{analysis_type} 타입이지만 표준 필드가 비어있음. 전체 데이터를 LLM에 전달")

                    # query, analysis_type 제외하고 나머지 데이터
                    filtered_data = {k: v for k, v in analysis_data.items()
                                   if k not in ['query', 'analysis_type'] and v}

                    if filtered_data:
                        analysis_text = json.dumps(filtered_data, ensure_ascii=False, indent=2)
                        logger.debug(f"전체 데이터 사용: {analysis_text[:200]}...")
                    else:
                        # 정말 데이터가 없으면 질문만이라도 LLM에 전달해서 일반 지식으로 답변하게
                        logger.warning("모든 데이터가 비어있음. LLM의 일반 지식으로 답변 생성 시도")
                        analysis_text = "정보 없음 - LLM의 일반 지식으로 답변 필요"

                # LLM이 정보를 정리하거나 일반 지식으로 답변
                prompt_template = llm_manager.get_prompt("report_direct_concept")
                prompt = prompt_template.format(
                    query=query,
                    analysis_text=analysis_text
                )

            else:
                logger.error(f"Unknown analysis_type: {analysis_type}")
                return f"❌ 지원하지 않는 분석 타입입니다: {analysis_type}"

            # LLM 직접 호출 (에이전트 없이)
            response = self.llm.invoke(prompt)
            report_text = response.content.strip()

            logger.info(f"보고서 생성 완료 - 길이: {len(report_text)} chars")
            return report_text

        except Exception as e:
            logger.error(f"직접 보고서 생성 실패: {str(e)}")
            return f"""# 보고서 생성 오류

보고서 생성 중 오류가 발생했습니다: {str(e)}

## 분석 데이터
```json
{json.dumps(analysis_data, ensure_ascii=False, indent=2)}
```

⚠️ 다시 시도해주세요.
"""


if __name__ == "__main__":
    import logging
    
    # 디버그 로그 활성화
    logging.getLogger("__main__").setLevel(logging.DEBUG)
    
    Config.validate_api_keys()
    
    # 테스트용 샘플 데이터
    SAMPLE_SINGLE_STOCK = {
        "analysis_type": "single",
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "current_price": 178.25,
        "analysis": "Apple continues to demonstrate strong fundamentals with robust iPhone sales and growing services revenue. The company's strategic focus on AI integration across its ecosystem, particularly with the introduction of Apple Intelligence features, positions it well for future growth. The services segment continues to show impressive growth, contributing significantly to overall revenue stability.",
        "metrics": {
            "pe_ratio": 29.5,
            "market_cap": 2800000000000,
            "52week_high": 199.62,
            "52week_low": 164.08,
            "sector": "Technology",
            "industry": "Consumer Electronics"
        },
        "period": "3mo",
        "news_summary": "Recent product launches have been well-received, with the iPhone 15 series showing strong demand in key markets. Apple's AI initiatives are gaining momentum, with developers showing significant interest in the new APIs and frameworks. The company's services ecosystem continues to expand with new offerings in financial services and health.",
        "analyst_recommendation": "Buy"
    }
    
    SAMPLE_COMPARISON = {
        "analysis_type": "comparison",
        "stocks": [
            {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "current_price": 178.25,
                "analysis": "Strong fundamentals with growing services revenue",
                "metrics": {
                    "pe_ratio": 29.5,
                    "market_cap": 2800000000000,
                    "52week_high": 199.62,
                    "52week_low": 164.08,
                    "sector": "Technology",
                    "industry": "Consumer Electronics"
                },
                "analyst_recommendation": "Buy"
            },
            {
                "ticker": "MSFT",
                "company_name": "Microsoft Corporation",
                "current_price": 380.50,
                "analysis": "Leading cloud and AI investments showing strong returns",
                "metrics": {
                    "pe_ratio": 35.2,
                    "market_cap": 2850000000000,
                    "52week_high": 425.00,
                    "52week_low": 309.45,
                    "sector": "Technology",
                    "industry": "Software"
                },
                "analyst_recommendation": "Strong Buy"
            }
        ],
        "comparison_analysis": "Both companies show strong fundamentals with Apple showing better value metrics (lower P/E) while Microsoft demonstrates stronger momentum (higher position in 52-week range). Microsoft's cloud and AI investments are showing stronger returns, while Apple's services growth provides more revenue stability.",
        "recommendation": {
            "preferred_stock": "MSFT",
            "reason": "Better positioned for AI-driven growth with Azure and enterprise cloud dominance",
            "risk_level": "Medium"
        }
    }
    
    # Report Generator 초기화
    print("\n" + "="*80)
    print("REPORT GENERATOR 테스트 시작")
    print("="*80)
    
    generator = ReportGenerator()
    
    # 출력 디렉토리 생성
    os.makedirs("charts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # ========================================================================
    # 시나리오 1: 차트만 요청
    # ========================================================================
    print("\n" + "="*80)
    print("시나리오 1: 차트만 요청 (단일 주식)")
    print("="*80)
    print("요청: '애플 주식 차트 그려줘'")
    print("-"*80)
    
    try:
        result = generator.generate_report(
            "애플 주식 차트 그려줘",
            SAMPLE_SINGLE_STOCK
        )
        print(f"\n[결과 상태]: {result['status']}")
        print(f"[생성된 차트]: {result['charts']}")
        print(f"\n[보고서 내용]:\n{result['report'][:500]}...")
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
    
    input("\n⏸️  Press Enter to continue to Scenario 2...")
    
    # ========================================================================
    # 시나리오 2: 분석만 요청
    # ========================================================================
    print("\n" + "="*80)
    print("시나리오 2: 분석만 요청")
    print("="*80)
    print("요청: '애플 주식 분석해줘'")
    print("-"*80)
    
    try:
        result = generator.generate_report(
            "애플 주식 분석해줘",
            SAMPLE_SINGLE_STOCK
        )
        print(f"\n[결과 상태]: {result['status']}")
        print(f"[생성된 차트]: {result['charts']}")
        print(f"[저장 경로]: {result['saved_path']}")
        print(f"\n[보고서 내용]:\n{result['report'][:800]}...")
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
    
    input("\n⏸️  Press Enter to continue to Scenario 3...")
    
    # ========================================================================
    # 시나리오 3: 저장 요청 (MD 형식)
    # ========================================================================
    print("\n" + "="*80)
    print("시나리오 3: 저장 요청 (MD 형식)")
    print("="*80)
    print("요청: '애플 주식 분석해서 저장해줘'")
    print("-"*80)
    
    try:
        result = generator.generate_report(
            "애플 주식 분석해서 저장해줘",
            SAMPLE_SINGLE_STOCK
        )
        print(f"\n[결과 상태]: {result['status']}")
        print(f"[생성된 차트]: {result['charts']}")
        print(f"[저장 경로]: {result['saved_path']}")
        print(f"\n[보고서 내용]:\n{result['report'][:800]}...")
        
        if result['saved_path']:
            print(f"\n✅ 파일 저장 성공: {result['saved_path']}")
        else:
            print("\n⚠️  파일 저장 실패 또는 저장 안 됨")
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
    
    input("\n⏸️  Press Enter to continue to Scenario 4...")
    
    # ========================================================================
    # 시나리오 4: 차트 + 저장 (PDF 형식)
    # ========================================================================
    print("\n" + "="*80)
    print("시나리오 4: 차트 + 저장 (PDF 형식)")
    print("="*80)
    print("요청: 'PDF로 차트 포함해서 저장해줘'")
    print("-"*80)
    
    try:
        result = generator.generate_report(
            "애플 주식 분석을 PDF로 차트 포함해서 저장해줘",
            SAMPLE_SINGLE_STOCK
        )
        print(f"\n[결과 상태]: {result['status']}")
        print(f"[생성된 차트]: {result['charts']}")
        print(f"[저장 경로]: {result['saved_path']}")
        print(f"\n[보고서 내용]:\n{result['report'][:800]}...")
        
        if result['saved_path'] and result['saved_path'].endswith('.pdf'):
            print(f"\n✅ PDF 파일 저장 성공: {result['saved_path']}")
            if result['charts']:
                print(f"   차트 포함됨: {', '.join(result['charts'])}")
        else:
            print("\n⚠️  PDF 저장 실패")
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
    
    input("\n⏸️  Press Enter to continue to Scenario 5...")
    
    # ========================================================================
    # 시나리오 5: 비교 분석 (AAPL vs MSFT)
    # ========================================================================
    print("\n" + "="*80)
    print("시나리오 5: 비교 분석 (AAPL vs MSFT)")
    print("="*80)
    print("요청: '애플과 마이크로소프트 비교 분석해서 차트와 함께 저장해줘'")
    print("-"*80)
    
    try:
        result = generator.generate_report(
            "애플과 마이크로소프트 비교 분석해서 차트와 함께 저장해줘",
            SAMPLE_COMPARISON
        )
        print(f"\n[결과 상태]: {result['status']}")
        print(f"[생성된 차트]: {result['charts']}")
        print(f"[저장 경로]: {result['saved_path']}")
        print(f"\n[보고서 내용]:\n{result['report'][:1000]}...")
        
        if result['charts']:
            print(f"\n✅ 비교 차트 생성됨: {', '.join(result['charts'])}")
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
    
    # ========================================================================
    # 테스트 요약
    # ========================================================================
    print("\n" + "="*80)
    print("테스트 완료!")
    print("="*80)
    print("\n📁 생성된 파일 확인:")
    print("  • charts/ 디렉토리")
    print("  • reports/ 디렉토리")
    print("\n💡 TIP: vector_search_agent 구현 후 시나리오 6 추가 예정")
