import logging

from functools import lru_cache
from langchain.tools import Tool
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.indices.base import BaseIndex

# 導入集中管理的服務與設定
from core.services import get_qdrant_client, configure_llama_index_settings, get_llama_gemini_flash
from core.config import settings

# 1. 連接到已存在的 Vector Store
@lru_cache(maxsize=None)
def get_vector_store():
    logging.info('首次初始化 QdrantVectorStore...')
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name="my_KnowledgeBase",
        enable_hybrid=True,
    )

# 2. 從 Vector Store 實例化索引物件
@lru_cache(maxsize=None)
def get_index() -> BaseIndex:
    logging.info('首次初始化 VectorStoreIndex...')
    configure_llama_index_settings()
    return VectorStoreIndex.from_vector_store(vector_store=get_vector_store())

# 3. 建立查詢引擎
def run_deep_research(query: str) -> str:
    logging.info("向量資料庫探索...")
    index = get_index()
    query_engine = index.as_query_engine(
                    llm = get_llama_gemini_flash(),
                    similarity_top_k=settings.SIMILARITY_TOP_K,
                    sparse_top_k=settings.SPARSE_TOP_K,
                    vector_store_query_mode=settings.QUERY_MODE,
                    alpha=settings.HYBRID_SEARCH_ALPHA,
                    num_queries = settings.NUM_QUERIES,
                    node_postprocessors = settings.POSTPROCESSORS,
                    streaming = settings.STREAMING,
                )
    result = query_engine.query(query)
    return str(result)

# 4. 定義並導出工具
DeepResearchKnowledgeBase = Tool(
    name="DeepResearchKnowledgeBase",
    func=run_deep_research,
    description="""一個深度研究工具，專門用於查詢和回答關於「臨床試驗設計」、「藥物開發流程」、以及「GCP規範」等專業領域的內部知識庫。
                 當使用者的問題涉及臨床試驗的具體執行細節、法規遵循或內部流程時，必須優先使用此工具。"""
)

if __name__ == "__main__":
    # 載入 .env 檔案中的環境變數 (API 金鑰等)
    # 請確保您已安裝 python-dotenv: pip install python-dotenv
    try:
        from dotenv import load_dotenv
        print("正在從 .env 檔案載入環境變數...")
        load_dotenv()
        print("載入完成。")
    except ImportError:
        print("未找到 python-dotenv，請確保您的 API 金鑰已手動設定為環境變數。")

    # 設定日誌，以便我們能看到函式內部的 INFO 訊息
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 定義一個符合工具描述的測試問題
    TEST_QUERY = "臨床試驗的 GCP 規範是什麼？"
    
    print("\n" + "="*50)
    print(f"🚀 開始驗證 DeepResearchKnowledgeBase 工具連線...")
    print(f"測試問題: {TEST_QUERY}")
    print("="*50 + "\n")

    try:
        # 直接呼叫核心函式，這會觸發所有相關的連線
        # 1. get_qdrant_client() -> 連線到 Qdrant
        # 2. configure_llama_index_settings() -> 初始化 Embedding 模型 (需要 Google API Key)
        # 3. get_llama_gemini_flash() -> 初始化 LLM (需要 Google API Key)
        result = run_deep_research(TEST_QUERY)
        
        print("\n" + "="*50)
        print("✅ 測試成功！所有連線均正常。")
        print("="*50)
        print("\n查詢結果預覽：")
        # 為了避免輸出過長，只顯示前 500 個字元
        print(result[:500] + "..." if len(result) > 500 else result)

    except Exception as e:
        print("\n" + "="*50)
        print("❌ 測試失敗！執行過程中發生錯誤。")
        print("="*50)
        # exc_info=True 會印出完整的錯誤追蹤回溯 (Traceback)
        # 這將是解決 UnexpectedResponse 的最關鍵線索
        logging.error("錯誤詳情如下:", exc_info=True)