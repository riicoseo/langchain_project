# src/workflow/nodes.py
"""
Workflow Nodes

LangGraph 워크플로우에서 사용되는 노드 함수들입니다.
"""

from typing import Dict, Any
from src.agents.financial_analyst import FinancialAnalyst
from src.agents.report_generator import ReportGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def financial_analyst_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    주식 분석 노드

    Args:
        state: 워크플로우 상태

    Returns:
        업데이트된 상태
    """
    logger.info("=== Financial Analyst Node 시작 ===")

    try:
        user_request = state.get("user_request", "")

        if not user_request:
            logger.error("user_request가 비어있습니다")
            state["error_message"] = "사용자 요청이 없습니다"
            return state

        # Financial Analyst 인스턴스 생성 및 실행
        analyst = FinancialAnalyst(model_name="solar-pro", temperature=0)
        analysis_data = analyst.analyze(user_request)

        # 결과 저장
        state["analysis_data"] = analysis_data

        logger.info(f"분석 완료 - type: {analysis_data.get('analysis_type', 'N/A')}")
        logger.info("=== Financial Analyst Node 완료 ===")

        return state

    except Exception as e:
        logger.error(f"Financial Analyst Node 오류: {str(e)}")
        state["error_message"] = f"분석 중 오류: {str(e)}"
        return state


def report_generator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    보고서 생성 노드

    request_type에 따라 다른 소스 사용:
    - "rag": rag/retriever.py에서 문서 검색
    - 그 외: state의 analysis_data 사용

    Args:
        state: 워크플로우 상태

    Returns:
        업데이트된 상태
    """
    logger.info("=== Report Generator Node 시작 ===")

    try:
        user_request = state.get("user_request", "")
        request_type = state.get("request_type", "")

        if not user_request:
            logger.error("user_request가 비어있습니다")
            state["error_message"] = "사용자 요청이 없습니다"
            return state

        # request_type에 따라 데이터 소스 결정
        if request_type == "rag":
            logger.info("RAG 모드: retriever에서 문서 검색")

            # TODO: rag/retriever.py 구현 후 활성화
            # from src.rag.retriever import Retriever
            # retriever = Retriever()
            # search_results = retriever.search(user_request, top_k=3)

            # 임시: RAG 더미 데이터
            analysis_data = {
                "source": "rag",
                "query": user_request,
                "documents": [
                    "PER(Price-to-Earnings Ratio)은 주가수익비율로, 주가를 주당순이익으로 나눈 값입니다.",
                    "PER이 낮을수록 저평가된 주식일 가능성이 높습니다."
                ],
                "search_results": "RAG 검색 결과 (더미 데이터)"
            }
            logger.warning("RAG 더미 데이터 사용 - 실제 구현 필요")
        else:
            logger.info(f"Financial 모드: analysis_data 사용 (request_type: {request_type})")
            analysis_data = state.get("analysis_data", {})

            if not analysis_data:
                logger.error("analysis_data가 비어있습니다")
                state["error_message"] = "분석 데이터가 없습니다"
                return state

        # Report Generator 인스턴스 생성 및 실행
        generator = ReportGenerator(model_name="solar-mini", temperature=0.3)
        report_result = generator.generate_report(user_request, analysis_data)

        # 결과 저장
        state["report"] = report_result.get("report", "")
        state["chart_paths"] = report_result.get("charts", [])
        state["saved_path"] = report_result.get("saved_path")

        logger.info(f"보고서 생성 완료 - 차트 개수: {len(state['chart_paths'])}")
        logger.info("=== Report Generator Node 완료 ===")

        return state

    except Exception as e:
        logger.error(f"Report Generator Node 오류: {str(e)}")
        import traceback
        logger.debug(f"상세 오류:\n{traceback.format_exc()}")
        state["error_message"] = f"보고서 생성 중 오류: {str(e)}"
        return state
