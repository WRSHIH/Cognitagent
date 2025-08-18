import argparse
import logging
import time
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams, Distance
from qdrant_client import models


from core.services import get_qdrant_client, LlamaSettings, configure_llama_index_settings
from core.config import settings

# 設定日誌記錄
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_ingestion(source_dir: str, collection_name: str, recreate: bool):
    """
    執行資料導入流程：讀取文件 -> 建立索引 -> 存入 Qdrant。

    Args:
        source_dir (str): 原始文件的來源資料夾路徑。
        collection_name (str): Qdrant 中的集合名稱。
        recreate (bool): 是否要刪除已存在的集合並重新建立。
    """
    logging.info(f"開始資料導入流程，來源: '{source_dir}'")
    logging.info("正在設定 LlamaIndex 全域參數...")
    configure_llama_index_settings()
    logging.info("LlamaIndex 設定完成。")


    if recreate:
        logging.warning(f"偵測到 --recreate 參數，準備刪除集合: '{collection_name}'")
        try:
            get_qdrant_client().delete_collection(collection_name=collection_name)
            logging.info(f"集合 '{collection_name}' 刪除成功。")
        except Exception as e:
            # 如果集合不存在，刪除會失敗，這是正常現象
            logging.warning(f"嘗試刪除集合 '{collection_name}' 失敗 (可能尚不存在): {e}")

        # 即使刪除失敗也要嘗試重新建立，以防萬一
        get_qdrant_client().recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            sparse_vectors_config={
                                    "text-sparse": models.SparseVectorParams(index=
                                                                             models.SparseIndexParams(on_disk=False,))}
                                                                             )
        logging.info(f"集合 '{collection_name}' 重新建立成功。")


    # 讀取來源資料夾中的文件
    logging.info("正在讀取文件...")
    reader = SimpleDirectoryReader(
        input_dir=source_dir,
        recursive=True,
        required_exts=[".pdf", ".json", ".md", ".txt"],
        exclude=["*.tmp"],
        encoding = 'utf-8'
    )
    documents = reader.load_data()
    logging.info(f"文件讀取完畢，共 {len(documents)} 份文件。")

    if not documents:
        logging.warning("來源資料夾中沒有找到任何可導入的文件，流程結束。")
        return

    
    logging.info("開始建立索引並將資料存入 Qdrant，此過程可能需要一些時間...")

    vector_store = QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=collection_name,
        enable_hybrid=True  # 啟用混合搜尋
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    chunk_size = 10
    for i in range(0, len(documents), chunk_size):
        doc_chunk = documents[i:i + chunk_size]
        logging.info(f"--- 正在處理第 {i // chunk_size + 1} 批文件 (文件索引 {i} 到 {i + len(doc_chunk) - 1}) ---")
        VectorStoreIndex.from_documents(
            doc_chunk,
            storage_context=storage_context,
            show_progress=False,
            llm=LlamaSettings.llm,
            embed_model=LlamaSettings.embed_model,
            node_parser=LlamaSettings.node_parser
        )
        logging.info(f"--- 第 {i // chunk_size + 1} 批文件處理完成 ---")

        if i + chunk_size < len(documents):
            logging.warning("API 速率限制：將暫停 61 秒，等待額度重置...")
            time.sleep(61)

    logging.info("✅ 資料導入流程成功完成！")

if __name__ == '__main__':
    # --- 命令列參數解析 ---
    parser = argparse.ArgumentParser(description="資料導入腳本，用於處理文件並存入 Qdrant 向量資料庫。")
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="包含原始文件的來源資料夾路徑 (例如: ./source_docs)。"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="my_collection",
        help="Qdrant 中要使用的集合名稱。"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="如果設置此參數，將會刪除並重新建立目標集合。"
    )
    args = parser.parse_args()

    # --- 執行主函式 ---
    run_ingestion(
        source_dir=args.source_dir,
        collection_name=args.collection,
        recreate=args.recreate
    )