# tests/core/test_services.py

import pytest
from unittest.mock import patch, MagicMock

# 導入我們要測試的目標模組
from core import services

# 使用 patch.object 來精準替換 services 模組中的 settings 物件
@patch.object(services.settings, 'GEMINI_FLASH', 'test-flash-model')
@patch.object(services.settings, 'GOOGLE_API_KEY')
@patch('core.services.GoogleGenAI')
def test_get_llm_flash_instantiation(mock_google_genai, mock_api_key):
    """
    測試 get_llama_gemini_flash 是否使用正確的設定來初始化 GoogleGenAI。
    """
    # 1. 準備 (Arrange)
    # 透過 patch.object，我們已經注入了假的 model_name。
    # 現在設定 mock_api_key 的 get_secret_value 回傳值。
    mock_api_key.get_secret_value.return_value = 'test-api-key'

    # 清除快取以確保獨立性
    services.get_llama_gemini_flash.cache_clear()

    # 2. 執行 (Act)
    services.get_llama_gemini_flash()

    # 3. 斷言 (Assert)
    # 驗證 GoogleGenAI 是否被以我們 patch 進去的值來呼叫
    mock_google_genai.assert_called_once_with(
        model_name="test-flash-model",
        api_key="test-api-key"
    )

@patch('core.services.GoogleGenAI')
def test_get_llm_flash_caching(mock_google_genai, monkeypatch):
    """
    測試 @lru_cache 是否有效。(此測試已通過，保持原樣)
    """
    # 為了讓此測試獨立，我們仍然需要設定環境
    monkeypatch.setenv("GEMINI_FLASH", "any-model")
    monkeypatch.setenv("GOOGLE_API_KEY", "any-key")
    import importlib
    from core import config
    importlib.reload(config)
    services.get_llama_gemini_flash.cache_clear()

    # 執行兩次
    instance1 = services.get_llama_gemini_flash()
    instance2 = services.get_llama_gemini_flash()

    # 斷言只被初始化一次
    mock_google_genai.assert_called_once()
    assert instance1 is instance2

# 【修正 Patch 路徑】
@patch('core.services.LlamaSettings')
@patch('core.services.UnstructuredElementNodeParser')
@patch('core.services.get_llama_gemini_embed')
@patch('core.services.get_llama_gemini_flash') # 使用您目前的正確函式名稱
def test_configure_llama_index_settings(
    mock_get_llm_flash,
    mock_get_embed_model,
    mock_unstructured_parser,
    mock_llama_settings
):
    """
    測試 configure_llama_index_settings 是否正確設定全域參數。
    """
    # 準備
    fake_llm = MagicMock()
    fake_embed_model = MagicMock()
    mock_get_llm_flash.return_value = fake_llm
    mock_get_embed_model.return_value = fake_embed_model
    
    # 執行
    services.configure_llama_index_settings()
    
    # 斷言
    assert mock_llama_settings.llm == fake_llm
    assert mock_llama_settings.embed_model == fake_embed_model
    mock_unstructured_parser.assert_called_once_with(llm=fake_llm)
    assert mock_llama_settings.node_parser == mock_unstructured_parser.return_value