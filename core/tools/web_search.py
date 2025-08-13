from langchain_community.tools.tavily_search import TavilySearchResults
from core.config import settings

search_tool = TavilySearchResults(
    max_results=settings.TAVILY_MAX_RESULTS,
    description="一個可以搜尋網路即時資訊的強大工具。當知識庫沒有答案，或問題涉及最新事件時使用。",
    tavily_api_key=settings.TAVILY_API_KEY.get_secret_value()
)