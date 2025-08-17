# tests/script/test_ingest.py

import pytest
from unittest.mock import patch, MagicMock

from script.ingest import run_ingestion

# 【修正 1】我們需要 patch LlamaSettings 來阻止真實模型被載入
@patch('script.ingest.LlamaSettings')
@patch('script.ingest.SimpleDirectoryReader')
@patch('script.ingest.VectorStoreIndex')
@patch('script.ingest.get_qdrant_client') 
def test_run_ingestion_success_no_recreate(
    mock_get_qdrant_client, mock_vector_store_index, mock_directory_reader, mock_llama_settings, tmp_path
):
    """
    測試成功情境：來源資料夾有文件，且不重建 collection。
    """
    # 準備
    source_dir = tmp_path / "source_docs"
    source_dir.mkdir()
    (source_dir / "doc1.md").write_text("文件一內容")
    mock_directory_reader.return_value.load_data.return_value = [MagicMock()]
    mock_client_instance = MagicMock()
    mock_get_qdrant_client.return_value = mock_client_instance

    # 執行
    run_ingestion(
        source_dir=str(source_dir),
        collection_name="test_collection",
        recreate=False
    )

    # 斷言
    mock_directory_reader.assert_called_once()
    mock_vector_store_index.from_documents.assert_called_once()
    mock_client_instance.delete_collection.assert_not_called()
    mock_client_instance.recreate_collection.assert_not_called()

@patch('script.ingest.LlamaSettings')
@patch('script.ingest.SimpleDirectoryReader')
@patch('script.ingest.VectorStoreIndex')
@patch('script.ingest.get_qdrant_client')
def test_run_ingestion_success_with_recreate(
    mock_get_qdrant_client, mock_vector_store_index, mock_directory_reader, mock_llama_settings, tmp_path
):
    """
    測試成功情境：使用 --recreate 參數。
    """
    # 準備
    source_dir = tmp_path / "source_docs"
    source_dir.mkdir()
    (source_dir / "doc1.pdf").write_text("PDF 內容")
    mock_directory_reader.return_value.load_data.return_value = [MagicMock()]
    mock_client_instance = MagicMock()
    mock_get_qdrant_client.return_value = mock_client_instance

    # 執行
    run_ingestion(
        source_dir=str(source_dir),
        collection_name="test_collection_recreate",
        recreate=True
    )

    # 斷言
    mock_client_instance.delete_collection.assert_called_once_with(collection_name="test_collection_recreate")
    mock_client_instance.recreate_collection.assert_called_once()
    mock_vector_store_index.from_documents.assert_called_once()

@patch('script.ingest.LlamaSettings')
@patch('script.ingest.SimpleDirectoryReader')
@patch('script.ingest.VectorStoreIndex')
@patch('script.ingest.get_qdrant_client')
def test_run_ingestion_empty_directory(
    mock_get_qdrant_client, mock_vector_store_index, mock_directory_reader, mock_llama_settings, tmp_path
):
    """
    測試邊界案例：來源資料夾為空。
    """
    # 準備
    source_dir = tmp_path / "empty_docs"
    source_dir.mkdir()
    mock_directory_reader.return_value.load_data.return_value = []
    mock_client_instance = MagicMock()
    mock_get_qdrant_client.return_value = mock_client_instance

    # 執行
    run_ingestion(
        source_dir=str(source_dir),
        collection_name="test_collection_empty",
        recreate=False
    )

    # 斷言
    mock_directory_reader.return_value.load_data.assert_called_once()
    mock_vector_store_index.from_documents.assert_not_called()