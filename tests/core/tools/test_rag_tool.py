# tests/core/tools/test_rag_tool.py

import pytest
from unittest.mock import patch, MagicMock

# 導入我們要測試的目標函式和工具
from core.tools.rag_tool import (
    get_vector_store,
    get_index,
    run_deep_research,
    DeepResearchKnowledgeBase,
)

# 測試 get_vector_store 的組裝邏輯
# 我們 patch 掉它依賴的兩個外部元件
@patch('core.tools.rag_tool.QdrantVectorStore')
@patch('core.tools.rag_tool.get_qdrant_client')
def test_get_vector_store(mock_get_qdrant_client, mock_qdrant_vector_store):
    """
    測試 get_vector_store 是否使用正確的參數初始化 QdrantVectorStore。
    """
    # 準備
    # 清除快取以確保測試獨立性
    get_vector_store.cache_clear()
    fake_client = MagicMock()
    mock_get_qdrant_client.return_value = fake_client

    # 執行
    vector_store = get_vector_store()

    # 斷言
    # 驗證 get_qdrant_client 被呼叫了一次
    mock_get_qdrant_client.assert_called_once()
    # 驗證 QdrantVectorStore 是否被以正確的參數初始化
    mock_qdrant_vector_store.assert_called_once_with(
        client=fake_client,
        collection_name="my_KnowledgeBase",
        enable_hybrid=True,
    )
    # 驗證回傳值是 mock 物件的實例
    assert vector_store == mock_qdrant_vector_store.return_value


# 測試 run_deep_research 的核心邏輯
@patch('core.tools.rag_tool.settings') # Patch settings 來控制參數
@patch('core.tools.rag_tool.get_index') # Patch get_index 來提供假的 index
def test_run_deep_research(mock_get_index, mock_settings):
    """
    測試 run_deep_research 是否使用正確的設定來建立查詢引擎並執行查詢。
    """
    # 1. 準備 (Arrange)
    # 準備一個假的查詢引擎，並設定它的 query 方法
    mock_query_engine = MagicMock()
    # 讓 query 方法回傳一個可以被 str() 轉換的假結果
    mock_query_engine.query.return_value = MagicMock(__str__=lambda self: "查詢成功")

    # 準備一個假的 index 物件，並設定它的 as_query_engine 方法
    mock_index_instance = MagicMock()
    mock_index_instance.as_query_engine.return_value = mock_query_engine
    
    # 設定 get_index 函式的回傳值
    mock_get_index.return_value = mock_index_instance

    # 準備假的設定值
    mock_settings.SIMILARITY_TOP_K = 99
    mock_settings.SPARSE_TOP_K = 88
    mock_settings.QUERY_MODE = "test_mode"
    mock_settings.HYBRID_SEARCH_ALPHA = 0.123
    mock_settings.NUM_QUERIES = 7
    mock_settings.POSTPROCESSORS = [MagicMock()]
    mock_settings.STREAMING = False

    # 2. 執行 (Act)
    result = run_deep_research("這是一個測試查詢")

    # 3. 斷言 (Assert)
    # 驗證 get_index 被呼叫了一次
    mock_get_index.assert_called_once()
    
    # 驗證 as_query_engine 是否被以我們設定的假參數正確地呼叫
    mock_index_instance.as_query_engine.assert_called_once_with(
        llm=mock_settings.GEMINI_FLASH, # 假設 llm 也是從 settings 來的
        similarity_top_k=99,
        sparse_top_k=88,
        vector_store_query_mode="test_mode",
        alpha=0.123,
        num_queries=7,
        node_postprocessors=mock_settings.POSTPROCESSORS,
        streaming=False,
    )
    
    # 驗證 query 方法是否被以正確的查詢字串呼叫
    mock_query_engine.query.assert_called_once_with("這是一個測試查詢")
    
    # 驗證最終回傳的結果是否正確
    assert result == "查詢成功"

def test_deep_research_tool_properties():
    """
    測試最終建立的 DeepResearchKnowledgeBase 工具物件的屬性是否正確。
    """
    # 斷言工具名稱
    assert DeepResearchKnowledgeBase.name == "DeepResearchKnowledgeBase"
    # 斷言工具的描述不是空的
    assert DeepResearchKnowledgeBase.description is not None
    assert len(DeepResearchKnowledgeBase.description) > 0
    # 斷言工具的執行函式是否指向我們測試的 run_deep_research
    assert DeepResearchKnowledgeBase.func == run_deep_research