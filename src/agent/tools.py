from langchain.tools import tool
import yfinance as yf
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

@tool
def get_stock_info(ticker: str) -> str:
    """주식의 상세 정보를 조회합니다."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 6개월 수익률 계산
        hist = stock.history(period="6mo")
        price_change_6m = 0
        if not hist.empty:
            price_change_6m = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
        
        result = f"""{info.get('longName', '')} ({ticker})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 현재 시장 정보:
• 현재가: ${info.get('currentPrice', 0):.2f}
• 전일 종가: ${info.get('previousClose', 0):.2f}
• 52주 최고: ${info.get('fiftyTwoWeekHigh', 0):.2f}
• 52주 최저: ${info.get('fiftyTwoWeekLow', 0):.2f}
• 6개월 수익률: {price_change_6m:.2f}%

💰 밸류에이션:
• PER: {info.get('trailingPE', 0):.2f}
• Forward PER: {info.get('forwardPE', 0):.2f}
• PEG Ratio: {info.get('pegRatio', 0):.2f}
• 시가총액: ${info.get('marketCap', 0):,}

📈 재무 지표:
• 배당수익률: {(info.get('dividendYield', 0) or 0) * 100:.2f}%
• 베타: {info.get('beta', 0):.2f}
• ROE: {(info.get('returnOnEquity', 0) or 0) * 100:.2f}%

🏢 기업 정보:
• 섹터: {info.get('sector', '')}
• 산업: {info.get('industry', '')}
        
⭐ 애널리스트 추천: {info.get('recommendationKey', 'none').upper()}
"""
        return result.strip()
    except Exception as e:
        return f"Error: {ticker} 정보를 가져올 수 없습니다. {str(e)}"


@tool
def search_stocks(query: str) -> str:
    """회사명, 키워드, 또는 산업으로 주식을 검색합니다.
    사용자가 티커를 모르거나 특정 분야의 주식을 찾을 때 사용합니다.
    
    Args:
        query: 검색어 (회사명, 산업, 키워드 등)
    
    Returns:
        검색된 주식 티커와 회사명 리스트
    """
    try:
        results = yf.Search(query, max_results=10)
        
        if not results.quotes:
            return f"'{query}'에 대한 검색 결과가 없습니다."
        
        output = f"""'{query}' 검색 결과:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"""
        for item in results.quotes:
            symbol = item['symbol']
            name = item.get('longname', item.get('shortname', ''))
            exchange = item.get('exchange', '')
            output += f"• {symbol} - {name} [{exchange}]\n"
        
        output += f"\n💡 상세 정보를 보려면 get_stock_info 도구를 사용하세요."
        return output.strip()
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def compare_stocks(tickers: str) -> str:
    """
    여러 주식을 비교 분석합니다. 가격, 밸류에이션, 성장성 등을 한눈에 비교할 수 있습니다.
    
    Args:
        tickers: 쉼표로 구분된 티커 리스트 (예: AAPL,MSFT,GOOGL)
    
    Returns:
        비교 분석 테이블 (가격, PER, 시가총액, 배당, 수익률 등)
    """
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        
        if len(ticker_list) < 2:
            return "비교하려면 최소 2개의 티커가 필요합니다."
        
        output = """주식 비교 분석
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"""
        
        comparison_data = []
        
        for ticker in ticker_list:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # 6개월 수익률
                hist = stock.history(period="6mo")
                price_change_6m = 0
                if not hist.empty:
                    price_change_6m = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                
                comparison_data.append({
                    'ticker': ticker,
                    'name': info.get('longName', '')[:30],
                    'price': info.get('currentPrice', 0),
                    'pe': info.get('trailingPE', 0),
                    'forward_pe': info.get('forwardPE', 0),
                    'market_cap': info.get('marketCap', 0),
                    'dividend': (info.get('dividendYield', 0) or 0) * 100,
                    'return_6m': price_change_6m,
                    'sector': info.get('sector', ''),
                    'recommendation': info.get('recommendationKey', 'none').upper(),
                })
            except Exception as e:
                output += f"⚠️ {ticker}: 데이터 조회 실패\n\n"
                continue
        
        # 비교 테이블 생성
        for data in comparison_data:
            output += f"""🏢 {data['ticker']} - {data['name']}
💵 현재가: ${data['price']:.2f}
📊 PER: {data['pe']:.2f} | Forward PER: {data['forward_pe']:.2f}
💰 시가총액: ${data['market_cap']:,}
💸 배당수익률: {data['dividend']:.2f}%
📈 6개월 수익률: {data['return_6m']:.2f}%
🏷️ 섹터: {data['sector']}
⭐ 추천: {data['recommendation']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        return output.strip()
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def convert_usd_to_krw(amount: float) -> str:
    """달러를 원화로 환전합니다.

    Args:
        amount: 환전할 달러 금액

    Returns:
        원화 금액과 환율 정보
    """
    try:
        # USD/KRW 환율 조회
        exchange_rate = yf.Ticker("KRW=X")
        rate_info = exchange_rate.info
        current_rate = rate_info.get('regularMarketPrice', 0)

        # 환율이 없을 경우 최근 종가 사용
        if current_rate == 0:
            hist = exchange_rate.history(period="1d")
            if not hist.empty:
                current_rate = hist['Close'].iloc[-1]

        krw_amount = amount * current_rate

        result = f"""💱 환율 계산
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💵 달러: ${amount:,.2f}
💴 원화: ₩{krw_amount:,.0f}
📊 환율: 1 USD = {current_rate:,.2f} KRW
"""
        return result.strip()
    except Exception as e:
        return f"Error: 환율 정보를 가져올 수 없습니다. {str(e)}"


# 도구 바인딩
tools = [
    get_stock_info,
    search_stocks,
    compare_stocks,
    convert_usd_to_krw,
]