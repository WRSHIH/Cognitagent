import logging
import json
import asyncio
from typing import Dict, Any, Type, List, Union, ClassVar # 👈 1. 從 typing 導入 ClassVar

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from core.services import get_langchain_gemini_pro

class CognitiveProcessorInput(BaseModel):
    task: str = Field(description="需要對上下文執行的具體任務描述...")
    context: Dict[str, Any] = Field(description="包含先前所有步驟結果的字典...")

class CognitiveProcessorTool(BaseTool):
    """
    一個用於內部認知處理的工具...
    """
    name: str = "CognitiveProcessorTool"
    description: str = (
        "當你需要對已經蒐集到的資訊（儲存在工作記憶體中）進行整理、分類、分組、排序、總結或提取關鍵點時使用此工具..."
    )

    # --- 最終修正：使用 ClassVar 標記 ---
    # 告訴 Pydantic 這是一個類別層級的設定，而不是一個需要驗證的模型欄位。
    pydantic_args_schema: ClassVar[Type[BaseModel]] = CognitiveProcessorInput

    # 為了滿足抽象類別的合約，我們必須實現 _run
    def _run(self, task: str, context: Dict[str, Any]) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._arun(task=task, context=context))
        
    async def _arun(self, task: str, context: Dict[str, Any]) -> str:
        # ... (此處的核心邏輯完全保持不變) ...
        logging.info(f"--- 認知處理工具：開始執行任務 '{task}' ---")
        if not context:
            return "錯誤：工作記憶體 (context) 為空，無法進行處理。"

        context_str = json.dumps(context, indent=2, ensure_ascii=False)
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "你是一位專業的資料分析師..."),
            ("human", """請根據以下上下文資訊...
             **上下文 (Context):**\n```json\n{context}\n```
             **任務 (Task):**\n{task}\n
             **你的處理結果:**""")
        ])
        chain = prompt_template | get_langchain_gemini_pro()

        try:
            response = await chain.ainvoke({"context": context_str, "task": task})
            logging.info("--- 認知處理工具：成功完成任務 ---")
            
            content: Union[str, List[Union[str, Dict]]] = response.content
            if isinstance(content, list):
                text_parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in content]
                return "\n".join(text_parts)
            
            return str(content)

        except Exception as e:
            logging.error(f"認知處理工具在執行時發生錯誤: {e}", exc_info=True)
            return f"在處理內部資料時發生錯誤: {e}"

# 導出工具的實例以供註冊
cognitive_processor_tool = CognitiveProcessorTool()