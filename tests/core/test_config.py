# tests/core/test_config.py

import pytest
from pydantic import ValidationError
import importlib

# 導入我們要測試的目標「類別」
from core.config import Settings

def test_settings_load_success(monkeypatch):
    """
    測試成功情境：當所有必要的環境變數都存在時，設定應能成功載入。
    """
    # 準備
    monkeypatch.setenv("GOOGLE_API_KEY", "fake_google_key")
    monkeypatch.setenv("QDRANT_API_KEY", "fake_qdrant_key")
    monkeypatch.setenv("TAVILY_API_KEY", "fake_tavily_key")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:1234")
    monkeypatch.setenv("GEMINI_FLASH", "models/gemini-1.0-pro")
    monkeypatch.setenv("GEMINI_PRO", "models/gemini-1.5-pro-latest")
    monkeypatch.setenv("GEMINI_EMBED", "models/text-embedding-004")

    # 執行：直接建立一個新的 Settings 實例來進行驗證
    settings = Settings() # pyright: ignore[reportCallIssue]

    # 斷言
    assert settings.GOOGLE_API_KEY.get_secret_value() == "fake_google_key"
    assert str(settings.QDRANT_URL) == "http://localhost:1234/"
    assert settings.HYBRID_SEARCH_ALPHA == 0.5

def test_settings_missing_required_variable(monkeypatch):
    """
    測試失敗情境：當一個必要的環境變數遺失時，應拋出 ValidationError。
    """
    # 1. 準備 (Arrange)
    # 我們明確地刪除 GOOGLE_API_KEY，確保它不在 os.environ 中
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    
    # 設定其他必要的變數
    monkeypatch.setenv("QDRANT_API_KEY", "fake_qdrant_key")
    monkeypatch.setenv("TAVILY_API_KEY", "fake_tavily_key")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:1234")
    monkeypatch.setenv("GEMINI_FLASH", "models/gemini-1.0-pro")
    monkeypatch.setenv("GEMINI_PRO", "models/gemini-1.5-pro-latest")
    monkeypatch.setenv("GEMINI_EMBED", "models/text-embedding-004")

    # 2. 執行與斷言 (Act & Assert)
    with pytest.raises(ValidationError) as excinfo:
        # 【終極修正】
        # 直接實例化 Settings，並傳入 _env_file=None 來禁止它讀取 .env 檔案。
        # 現在，Settings 只能從我們用 monkeypatch 控制的 os.environ 中讀取設定。
        Settings(_env_file=None) # pyright: ignore[reportCallIssue]
    
    # 驗證錯誤訊息中是否包含了遺失的欄位名稱
    assert "GOOGLE_API_KEY" in str(excinfo.value)