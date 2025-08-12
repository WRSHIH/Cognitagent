from pathlib import Path
import logging

# 建立一個指向專案根目錄的絕對路徑

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"

def load_prompt(file_name: str) -> str:
    """
    從 prompts 資料夾安全地載入一個 Prompt 模板。

    Args:
        file_name (str): 要載入的 prompt 檔案名稱 (例如 "knowledge_merger.txt")。

    Returns:
        str: 檔案的內容。
    """
    prompt_path = PROMPTS_DIR / file_name
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"Prompt 檔案未找到: {prompt_path}")
        return f"錯誤: Prompt 檔案 '{file_name}' 未找到。"
    except Exception as e:
        logging.error(f"讀取 Prompt 檔案時發生錯誤 {prompt_path}: {e}")
        return f"錯誤: 讀取 Prompt 檔案 '{file_name}' 時發生錯誤。"