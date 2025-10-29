from langchain.tools import tool
from langchain_tavily import TavilySearch


@tool
def write_file(filename: str, content: str) -> str:
    """
    내용(content)을 지정된 파일 이름(filename)으로 저장합니다. 
    분석 결과나 검색 내용을 저장할 때 사용합니다.
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"✅ 성공: '{filename}' 파일이 성공적으로 저장되었습니다."
    except Exception as e:
        return f"❌ 오류: 파일을 저장하는 중 문제가 발생했습니다. {str(e)}"

@tool
def web_search(query: str) -> str:
    """
    최신 뉴스, 사건, 특정 주제에 대한 정보를 웹에서 검색합니다. 
    주가 변동의 '이유'를 찾거나, yfinance에 없는 정보를 검색할 때 사용합니다.
    """
    try:
        tavily = TavilySearch(max_results=3)
        results = tavily.invoke(query)
        
        # 검색 결과를 보기 좋게 정리
        output = f"🌐 '{query}'에 대한 웹 검색 결과:\n"
        output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        if not results:
            return "검색 결과가 없습니다."
            
        for res in results:
            output += f"🔗 출처: {res.get('url', 'N/A')}\n"
            output += f"📝 내용: {res.get('content', 'N/A')}\n\n"
            
        return output.strip()
    except Exception as e:
        return f"❌ 오류: 웹 검색 중 문제가 발생했습니다. {str(e)}"