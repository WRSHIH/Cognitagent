import logging
import json
import asyncio
from typing import Dict, Any, Type, List, Union, ClassVar

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from core.services import get_langchain_gemini_pro, get_langchain_gemini_flash_lite

CONTEXT_COMPRESSION_THRESHOLD = 15000

class CognitiveProcessorInput(BaseModel):
    task: str = Field(description="需要對上下文執行的具體任務描述...")
    context: Dict[str, Any] = Field(description="包含先前所有步驟結果的字典...")

class CognitiveProcessorTool(BaseTool):
    name: str = "CognitiveProcessorTool"
    description: str = (
        "當你需要對已經蒐集到的資訊（儲存在工作記憶體中）進行整理、分類、分組、排序、總結或提取關鍵點時使用此工具..."
    )

    pydantic_args_schema: ClassVar[Type[BaseModel]] = CognitiveProcessorInput

    async def _compress_context_if_needed(self, context_str: str, task: str) -> str:
        """如果上下文過長，則使用輕量級 LLM 進行摘要壓縮。"""
        if len(context_str) <= CONTEXT_COMPRESSION_THRESHOLD:
            logging.info("--- 認知處理工具：上下文長度在安全範圍內，無需壓縮。 ---")
            return context_str

        logging.warning(f"--- 認知處理工具：上下文長度 ({len(context_str)}) 超過閾值 ({CONTEXT_COMPRESSION_THRESHOLD})，正在啟動壓縮程序... ---")
        
        compression_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一位高效的資訊壓縮專家。你的任務是將提供的上下文濃縮成一個更短的版本，同時保留所有與後續任務相關的關鍵事實、數據和實體。"),
            ("human", """請壓縮以下上下文，使其更簡潔，以便後續執行這個任務：'{task}'
            
            **原始上下文 (Original Context):**
            ```json
            {context}
            ```

            **壓縮後的摘要 (Compressed Summary):**""")
        ])
        
        summarizer_chain = compression_prompt | get_langchain_gemini_flash_lite()
        
        try:
            response = await summarizer_chain.ainvoke({"context": context_str, "task": task})
            compressed_context = str(response.content)
            print(f"compressed_context: {response}")
            logging.info(f"--- 認知處理工具：上下文成功壓縮，新長度為 {len(compressed_context)} ---")
            return compressed_context
        except Exception as e:
            logging.error(f"--- 認知處理工具：上下文壓縮失敗: {e}。將使用原始上下文繼續，可能會有風險。 ---")
            return context_str # 如果壓縮失敗，回傳原始上下文

    def _run(self, task: str, context: Dict[str, Any]) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._arun(task=task, context=context))

    async def _arun(self, task: str, context: Dict[str, Any]) -> str:
        logging.info(f"--- 認知處理工具：開始執行任務 '{task}' ---")
        if not context:
            return "錯誤：工作記憶體 (context) 為空，無法進行處理。"

        context_str = json.dumps(context, indent=2, ensure_ascii=False)
        processed_context_str = await self._compress_context_if_needed(context_str, task)

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "你是一位專業的資料分析師，你的任務是根據提供的上下文和具體任務，產出精確、結構化的分析結果。請直接輸出結果，不要包含任何額外的解釋或開場白。"),
            ("human", """請根據以下上下文資訊，嚴格執行指定任務。
             **上下文 (Context):**
             ```json
             {context}
             ```
             **任務 (Task):**
             {task}

             **你的處理結果:**""")
        ])
        primary_chain = prompt_template | get_langchain_gemini_pro()

        try:
            response = await primary_chain.ainvoke({"context": processed_context_str, "task": task})
            print(f"主要回應(PRO): {response}")
            logging.info("--- 認知處理工具：成功完成主要任務 ---")
            content: Union[str, List[Union[str, Dict]]] = response.content
            if isinstance(content, list):
                text_parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in content]
                return "\n".join(text_parts)
            return str(content)

        except ValueError as e:
            # 專門捕捉 "No generations found in stream" 錯誤並啟用備用方案
            if "No generations found in stream" in str(e):
                logging.warning(f"--- 認知處理工具：主要模型因內容審核未回傳結果。正在啟用備用方案 (Fallback)... ---")
                
                # 備用的、更簡單的 Prompt
                fallback_prompt_template = ChatPromptTemplate.from_messages([
                    ("system", "你是一位資料摘要專家。請根據上下文，總結出最重要的核心資訊。"),
                    ("human", """請簡潔地總結以下資訊的關鍵重點。
                     **上下文 (Context):**
                     ```json
                     {context}
                     ```
                     **重點摘要:**""")
                ])
                fallback_chain = fallback_prompt_template | get_langchain_gemini_flash_lite()
                
                try:
                    fallback_response = await fallback_chain.ainvoke({"context": processed_context_str})
                    print(f"備用回應(LITE): {fallback_response}")
                    logging.info("--- 認知處理工具：備用方案成功完成 ---")
                    return str(fallback_response.content)
                except Exception as fallback_e:
                    error_message = f"FATAL: 認知處理工具的主要方案與備用方案皆執行失敗。備用方案錯誤: {fallback_e}"
                    logging.error(error_message, exc_info=True)
                    return error_message
            else:
                # 如果是其他 ValueError，則按原樣拋出
                logging.error(f"認知處理工具在執行時發生未預期的 ValueError: {e}", exc_info=True)
                return f"在處理內部資料時發生錯誤: {e}"
        except Exception as e:
            logging.error(f"認知處理工具在執行時發生一般性錯誤: {e}", exc_info=True)
            return f"在處理內部資料時發生錯誤: {e}"

# 導出工具的實例以供註冊
cognitive_processor_tool = CognitiveProcessorTool()