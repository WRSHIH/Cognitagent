import datetime
import json
import logging
from typing import List, Dict, Any

from langchain.tools import Tool
from llama_index.core import Document

# 從我們的工具函式庫中導入新的輔助函式
from core.utils import load_prompt

# 導入 rag_tool 中已經建立好的 index 物件，以供重複使用
from .rag_tool import get_index

# 導入集中管理的 LLM 服務與設定
from core.services import get_llama_gemini_flash
from core.config import settings

# --- 使用輔助函式載入 Prompt，程式碼更簡潔 ---
MERGER_PROMPT_TEMPLATE = load_prompt("knowledge_merger.txt")
ATOMIZER_PROMPT_TEMPLATE = load_prompt("knowledge_atomizer.txt")


def merge_knowledge_with_llm(old_text: str, new_text: str) -> str:
    """
    使用 LLM 智能合併新舊兩份知識。

    Args:
        old_text: 資料庫中已存在的舊知識文本。
        new_text: 準備寫入的新知識文本。
        llm: 用於執行的語言模型。

    Returns:
        合併後的、更完整準確的知識文本。
    """
    merger_prompt = MERGER_PROMPT_TEMPLATE.format(old_text=old_text, new_text=new_text)
    try:
        response = get_llama_gemini_flash().complete(merger_prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"智能合併時發生錯誤：{e}")
        return old_text

def atomize_knowledge(text_block: str) -> List[Dict[str, Any]]:
    """
    使用 LLM 將一段文本塊拆解成多個原子知識單元。每個單元包含文本和豐富的元數據。
    Args:
        text_block: 要進行原子化處理的文本塊。
    Returns:
        List[Dict[str, Any]]: 包含多個原子知識單元，每個單元是一個字典，包含文本、類型、關鍵詞和問題。
    例如：
        [
            {
                "text": "這是一個原子知識單元。",
                "type": "Fact",
                "keywords": ["原子", "知識"],
                "question": "這個原子知識單元是什麼？"
            },
            ...
        ]
    注意：此處的 llm 物件是從外部導入的，不再需要作為參數傳入。
    這樣可以確保我們使用的是集中管理的 LLM 實例，避免重複創建。 
    """
    atomizer_prompt = ATOMIZER_PROMPT_TEMPLATE.format(text_block=text_block)
    try:
        response = get_llama_gemini_flash().complete(atomizer_prompt)
        # 移除 ```json 和 ```
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response)
    except Exception as e:
        logging.error(f"知識原子化失敗: {e}")
        return [{"text": text_block, "type": "Unprocessed", "keywords": [], "question": ""}]

def save_new_knowledge(knowledge_text: str) -> str:
    """
    接收新知識，原子化後，與知識庫比對並進行新增或智能更新。
    注意：此處的 llm 和 index 物件都是從外部導入的，不再需要作為參數傳入。
    Args:
        knowledge_text: 要保存的新知識文本。
    Returns:
        str: 最終的確認消息，表示知識已成功保存或更新。
    注意：此函式會自動處理知識的原子化、合併和保存過程，
        並且會使用導入的 llm 和 index 物件，
        無需在函式調用時傳入這些參數。  
    """
    logging.info("開始執行知識進化流程...")
    atomic_units = atomize_knowledge(knowledge_text)
    if not atomic_units or atomic_units[0].get("type") == "Unprocessed":
        return "知識原子化失敗，未儲存任何內容。"
    logging.info(f"原子化後的知識單元數量: {len(atomic_units)}")

    SIMILARITY_THRESHOLD = settings.KNOWLEDGE_SIMILARITY_THRESHOLD  # 從設定中獲取相似度閾值
    retriever = get_index().as_retriever(similarity_top_k=1)    # 建立搜索器 確認知識內容是否已存在
    nodes_to_delete = set()                 # 使用 set 來追蹤需要刪除的 node_id，避免重複刪除
    documents_to_insert = []                # 紀錄需要更新的文件
    stats = {"new": 0, "updated": 0, "skipped": 0}

    for unit in atomic_units:
        new_text = unit.get("text", "")
        if not new_text:
            continue

        existing_nodes = retriever.retrieve(new_text)
        # 優化: 將 is_duplicate 檢查合併成一行
        is_duplicate = existing_nodes and existing_nodes[0].get_score() > SIMILARITY_THRESHOLD

        if is_duplicate:
            old_node = existing_nodes[0]
            old_text = old_node.get_text()

            logging.info(f"發現潛在可更新知識 (分數: {old_node.get_score():.4f})。")
            consolidated_text = merge_knowledge_with_llm(old_text, new_text)

            if consolidated_text.strip() != old_text.strip():
                logging.info(f"  [舊]: {old_text}")
                logging.info(f"  [新]: {new_text}")
                logging.info(f"  [合併後]: {consolidated_text}")
                nodes_to_delete.add(old_node.node_id)
                metadata = old_node.metadata | {
                    "last_modified_at": datetime.datetime.now().isoformat(),
                    "update_source": "LLM-Merge"
                }
                documents_to_insert.append(Document(text=consolidated_text, metadata=metadata))
                stats["updated"] += 1
            else:
                logging.info("無實質變化，跳過操作。")
                stats["skipped"] += 1
        else:
            logging.info(f"判定為全新知識，準備寫入: {new_text}")
            metadata = {
                "source": "AI-Generated via ReAct Agent",
                "creation_date": datetime.datetime.now().isoformat(),
                "knowledge_type": unit.get("type", "N/A"),
                "keywords": ", ".join(unit.get("keywords", [])),
                "potential_question": unit.get("question", "")
            }
            documents_to_insert.append(Document(text=new_text, metadata=metadata))
            stats["new"] += 1

    try:
        if nodes_to_delete:
            logging.info(f"正在從資料庫中刪除 {len(nodes_to_delete)} 個過時節點...")
            get_index().delete_nodes(list(nodes_to_delete))
        if documents_to_insert:
            logging.info(f"正在向資料庫中寫入 {len(documents_to_insert)} 個新/更新節點...")
            get_index().insert_nodes(documents_to_insert)

        final_message = (
            f"知識進化流程完成！\n"
            f"  - 新增知識: {stats['new']} 條\n"
            f"  - 更新知識: {stats['updated']} 條\n"
            f"  - 忽略知識: {stats['skipped']} 條"
        )
        logging.info(final_message)
        return final_message
    except Exception as e:
        error_message = f"在執行資料庫操作時發生嚴重錯誤：{e}"
        logging.info(error_message)
        return error_message


# 定義並導出工具
SaveNewKnowledgeTool = Tool(
    name="SaveNewKnowledgeTool",
    func=save_new_knowledge,
    description="當你合成了一個新的、且經過使用者確認的知識後，使用此工具將其保存到核心知識庫中。輸入參數應該是要保存的完整知識文本。"
)