from src.agent.tools import get_stock_info, search_stocks, compare_stocks, convert_usd_to_krw

print("=== Tool Test ===\n")

# Test 1
print("1. get_stock_info('AAPL')")
print(get_stock_info.invoke({"ticker": "AAPL"}))
print("\n" + "="*80 + "\n")

# Test 2
print("2. search_stocks('tesla')")
print(search_stocks.invoke({"query": "tesla"}))
print("\n" + "="*80 + "\n")

# Test 3
print("3. compare_stocks('AAPL,MSFT')")
print(compare_stocks.invoke({"tickers": "AAPL,MSFT"}))
print("\n" + "="*80 + "\n")

# Test 4
print("4. convert_usd_to_krw(100)")
print(convert_usd_to_krw.invoke({"amount": 100}))
print("\n" + "="*80 + "\n")