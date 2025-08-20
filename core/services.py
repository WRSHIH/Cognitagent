from functools import lru_cache
from llama_index.core import Settings as LlamaSettings
from llama_index.core.node_parser import UnstructuredElementNodeParser
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import qdrant_client
import logging

# 從我們的設定檔中導入 settings 物件
from .config import settings

# --- LLM 和 Embedding 服務初始化 ---

# 初始化 LLM
@lru_cache(maxsize=None)
def get_llama_gemini_flash():
    logging.info('首次初始化 Gemini 2.5 Flash...')
    return GoogleGenAI(
        model_name=settings.GEMINI_FLASH,
        api_key=settings.GOOGLE_API_KEY.get_secret_value()
    )

@lru_cache(maxsize=None)
def get_langchain_gemini_pro():
    logging.info('首次初始化 Gemini 2.5 Pro...')
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_PRO,
        api_key=settings.GOOGLE_API_KEY.get_secret_value(),
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
    )

# 初始化 Embedding 模型
@lru_cache(maxsize=None)
def get_llama_gemini_embed():
    logging.info('首次初始化 Embedding Model...')
    return GoogleGenAIEmbedding(
        model_name=settings.GEMINI_EMBED,
        api_key=settings.GOOGLE_API_KEY.get_secret_value(),
        embedding_config=types.EmbedContentConfig(output_dimensionality=settings.GEMINI_DIMENSION),
        task_type="RETRIEVAL_DOCUMENT",
        embed_batch_size=1,
    )

# --- LlamaIndex 全域背景設定 ---
def configure_llama_index_settings():
    logging.info("設定 LlamaIndex 全域參數...")
    LlamaSettings.llm = get_llama_gemini_flash()
    LlamaSettings.embed_model = get_llama_gemini_embed()
    LlamaSettings.node_parser = UnstructuredElementNodeParser(llm=get_llama_gemini_flash())
    


# --- 外部服務客戶端初始化 ---
@lru_cache(maxsize=None)
def get_qdrant_client():
    logging.info("首次初始化 Qdrant Client...")
    return qdrant_client.QdrantClient(
        url=str(settings.QDRANT_URL), # HttpUrl 類型需轉為字串
        api_key=settings.QDRANT_API_KEY.get_secret_value()
    )


if __name__ == '__main__':
    # from services import get_llama_gemini_embed
    embeddings = get_llama_gemini_embed().get_text_embedding("Google Gemini Embeddings.")
    print(f"生成的向量維度: {len(embeddings)}")
    
