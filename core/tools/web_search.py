from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from core.config import settings


search_tool = GoogleSerperRun(name="google_serper_search",
                              api_wrapper=GoogleSerperAPIWrapper(serper_api_key=settings.SERPER_API_KEY.get_secret_value()),
                              description="一個可以搜尋網路即時資訊的強大工具。當知識庫沒有答案，或問題涉及最新事件、人物、地點或通用事實時使用。")

# print(search_tool.run('鐵達尼號的導演是誰'))