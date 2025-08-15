import logging

from functools import lru_cache
from langchain.tools import Tool
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.indices.base import BaseIndex

# 導入集中管理的服務與設定
from core.services import get_qdrant_client, configure_llama_index_settings
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
                    llm = settings.GEMINI_FLASH,
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
    description="""一個深度研究工具，專門用於回答關於個人知識庫中複雜、模糊或需要從多個角度探討的問題。
                        此工具內部會自動生成多個子問題進行探索式查詢，並結合了關鍵字和語意搜尋。
                        當使用者的問題比較籠統，或暗示需要全面性的答案時，應優先使用此工具。"""
)