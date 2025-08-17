# tests/core/test_utils.py

import pytest
from unittest.mock import patch, mock_open

# 導入我們要測試的目標函式
from core.utils import load_prompt


# tests/core/test_utils.py

import pytest
from unittest.mock import patch, mock_open

from core.utils import load_prompt

# 模擬的 PROMPTS_DIR 路徑，這樣測試就不依賴真實的檔案結構
# 我們 patch 'core.utils.PROMPTS_DIR' 來控制函式內部的路徑變數
@patch('core.utils.PROMPTS_DIR')
def test_load_prompt_success(mock_prompts_dir):
    """
    測試成功情境：當檔案存在時，應回傳其內容。
    """
    # 1. 準備 (Arrange)
    fake_file_content = "這是一個提示詞模板。"
    # 使用 mock_open 來模擬一個已經被開啟、且內容為 fake_file_content 的檔案
    m = mock_open(read_data=fake_file_content)
    
    # 使用 patch 來攔截 builtins.open (即全域的 open 函式)
    # 當程式碼中呼叫 open(...) 時，實際上會呼叫我們偽造的 m
    with patch('builtins.open', m):
        # 2. 執行 (Act)
        result = load_prompt("any_file.txt")
        
        # 3. 斷言 (Assert)
        # 驗證回傳的結果是否與我們偽造的內容一致
        assert result == fake_file_content
        # 驗證 open 函式是否被以正確的路徑和參數呼叫
        m.assert_called_once()


@patch('core.utils.PROMPTS_DIR')
def test_load_prompt_file_not_found(mock_prompts_dir):
    """
    測試失敗情境一：當檔案不存在時，應回傳指定的錯誤訊息。
    """
    # 1. 準備 (Arrange)
    # 這次，我們讓 mock_open 在被呼叫時，直接拋出 FileNotFoundError
    m = mock_open()
    m.side_effect = FileNotFoundError

    with patch('builtins.open', m):
        # 2. 執行 (Act)
        result = load_prompt("non_existent_file.txt")
        
        # 3. 斷言 (Assert)
        assert "錯誤: Prompt 檔案 'non_existent_file.txt' 未找到。" in result

@patch('core.utils.PROMPTS_DIR')
def test_load_prompt_other_exception(mock_prompts_dir):
    """
    測試失敗情境二：當發生其他讀取錯誤時，應回傳通用的錯誤訊息。
    """
    # 1. 準備 (Arrange)
    # 模擬一個通用的 Exception，例如權限錯誤
    m = mock_open()
    m.side_effect = IOError("Permission denied")

    with patch('builtins.open', m):
        # 2. 執行 (Act)
        result = load_prompt("any_file.txt")
        
        # 3. 斷言 (Assert)
        assert "錯誤: 讀取 Prompt 檔案 'any_file.txt' 時發生錯誤。" in result