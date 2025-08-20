from dotenv import load_dotenv
from pydantic import SecretStr, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from llama_index.core.postprocessor import LongContextReorder

# 在所有程式碼執行之前，先執行 load_dotenv()
# 這會將 .env 檔案中的鍵值對載入到系統的環境變數中
# 如果環境中已存在同名變數，預設不會覆寫，這正是我們想要的行為
load_dotenv()

# 告訴 Pydantic 從 .env 檔案讀取設定
class Settings(BaseSettings):
    """
    應用程式的集中設定管理類別。
    Pydantic 會自動從環境變數或 .env 檔案中讀取對應的欄位。
    """
    
    # 必要金鑰 (Secrets) 
    GOOGLE_API_KEY: SecretStr
    QDRANT_API_KEY: SecretStr
    TAVILY_API_KEY: SecretStr

    # 資料庫與服務設定
    QDRANT_URL: HttpUrl

    # 模型設定
    GEMINI_FLASH: str
    GEMINI_PRO: str
    GEMINI_EMBED: str
    GEMINI_DIMENSION:int = 1024


    # llamaindex 檢索相關設定
    HYBRID_SEARCH_ALPHA: float = 0.5
    SIMILARITY_TOP_K: int = 5
    SPARSE_TOP_K: int = 5
    QUERY_MODE: str = "hybrid"
    NUM_QUERIES:int = 4
    POSTPROCESSORS: list = [LongContextReorder()]
    STREAMING: bool = True

    # Qdrant 檢索相關設定
    DENSE_VECTOR_SIZE: int = 1024

    # 知識回寫相關設定
    KNOWLEDGE_SIMILARITY_THRESHOLD: float = 0.95

    # 網路搜尋相關設定
    TAVILY_MAX_RESULTS: int = 3

    # Pydantic V2 的設定方式，指定 .env 檔案的編碼
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', case_sensitive=False)


# 建立一個全域可用的設定實例
# 應用程式的其他部分應該 import 這個 settings 物件來使用
settings = Settings() # type: ignore