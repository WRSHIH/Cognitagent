from llama_index.core import Settings as LlamaSettings
from llama_index.core.node_parser import UnstructuredElementNodeParser
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import qdrant_client

# 從我們的設定檔中導入 settings 物件
from .config import settings

# --- LLM 和 Embedding 服務初始化 ---

# 初始化 LLM
llm_flash = GoogleGenAI(
    model_name=settings.GEMINI_FLASH,
    api_key=settings.GOOGLE_API_KEY.get_secret_value()
)

llm_pro = GoogleGenAI(
    model_name=settings.GEMINI_PRO,
    api_key=settings.GOOGLE_API_KEY.get_secret_value()
)


# 初始化 Embedding 模型
embed_model = GoogleGenAIEmbedding(
    model_name=settings.GEMINI_EMBED,
    api_key=settings.GOOGLE_API_KEY.get_secret_value(),
    task_type="RETRIEVAL_DOCUMENT"
)

# --- LlamaIndex 全域背景設定 ---
LlamaSettings.llm = llm_flash
LlamaSettings.embed_model = embed_model
LlamaSettings.node_parser = UnstructuredElementNodeParser(llm=llm_flash)


# --- 外部服務客戶端初始化 ---
# 初始化 Qdrant 客戶端
qdrant_client_instance = qdrant_client.QdrantClient(
    url=str(settings.QDRANT_URL), # HttpUrl 類型需轉為字串
    api_key=settings.QDRANT_API_KEY.get_secret_value()
)

print("✅ Services initialized and LlamaIndex settings configured successfully.")