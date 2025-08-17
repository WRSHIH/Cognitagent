# tests/core/test_services.py

import pytest
from unittest.mock import patch, MagicMock

# 導入我們要測試的目標函式和模組
from core import services
from core.config import Settings


@patch('core.services.GoogleGenAI')
def test_get_llm_flash_instantiation(mock_google_genai, monkeypatch):
    """
    測試場景：get_llm_flash 函式是否使用正確的設定來初始化 GoogleGenAI。
    """
    # 1. 準備 (Arrange)
    # 使用 monkeypatch 來設定一個假的環境，以控制 settings 的值
    monkeypatch.setenv("GEMINI_FLASH", "test-flash-model")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
    # 為了讓 settings 重新讀取我們設定的假環境，需要重新載入 config 模組
    import importlib
    from core import config
    importlib.reload(config)

    # 在每次測試前清除快取，確保測試的獨立性
    services.get_llama_gemini_flash.cache_clear()

    # 2. 執行 (Act)
    llm_instance = services.get_llama_gemini_flash()

    # 3. 斷言 (Assert)
    # 驗證 GoogleGenAI 這個類別是否被以正確的參數呼叫了一次
    mock_google_genai.assert_called_once_with(
        model_name="test-flash-model",
        api_key="test-api-key"
    )
    # 驗證函式的回傳值就是 mock 類別的回傳實例
    assert llm_instance == mock_google_genai.return_value

@patch('core.services.GoogleGenAI')
def test_get_llm_flash_caching(mock_google_genai, monkeypatch):
    """
    測試場景：驗證 @lru_cache 是否有效，重複呼叫 get_llm_flash 時，
              GoogleGenAI 的初始化只會發生一次。
    """
    # 準備
    monkeypatch.setenv("GEMINI_FLASH", "any-model")
    monkeypatch.setenv("GOOGLE_API_KEY", "any-key")
    import importlib
    from core import config
    importlib.reload(config)
    services.get_llama_gemini_flash.cache_clear()

    # 執行：連續呼叫兩次
    instance1 = services.get_llama_gemini_flash()
    instance2 = services.get_llama_gemini_flash()

    # 斷言
    # 驗證 GoogleGenAI 的建構函式【只被呼叫了一次】
    mock_google_genai.assert_called_once()
    # 驗證兩次回傳的是同一個物件實例
    assert instance1 is instance2

@patch('core.services.UnstructuredElementNodeParser')
@patch('core.services.get_embed_model')
@patch('core.services.get_llm_flash')
@patch('core.services.LlamaSettings') # 也 patch LlamaSettings 本身
def test_configure_llama_index_settings(
    mock_llama_settings,
    mock_get_llm_flash,
    mock_get_embed_model,
    mock_unstructured_parser
):
    """
    測試場景：configure_llama_index_settings 是否將 get_* 函式的回傳值
              正確地設定到 LlamaSettings 的屬性上。
    """
    # 1. 準備 (Arrange)
    # 為我們的 get_* mock 函式準備好假的回傳值
    fake_llm = MagicMock()
    fake_embed_model = MagicMock()
    mock_get_llm_flash.return_value = fake_llm
    mock_get_embed_model.return_value = fake_embed_model
    
    # 2. 執行 (Act)
    services.configure_llama_index_settings()
    
    # 3. 斷言 (Assert)
    # 驗證 LlamaSettings 的屬性是否被賦值為我們準備的假物件
    assert mock_llama_settings.llm == fake_llm
    assert mock_llama_settings.embed_model == fake_embed_model
    # 驗證 UnstructuredElementNodeParser 是否用我們的假 llm 來初始化
    mock_unstructured_parser.assert_called_once_with(llm=fake_llm)
    # 驗證 LlamaSettings.node_parser 是否被賦值為 parser 的實例
    assert mock_llama_settings.node_parser == mock_unstructured_parser.return_value

