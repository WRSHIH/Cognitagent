from pathlib import Path
import itertools

# --- 設定 (Configuration) ---
# 將所有設定集中在此，方便修改
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIRS = ["app", "core", "script"]
TESTS_ROOT_NAME = "tests"
EXCLUDE_FILES = {"__init__.py"} 

def create_test_structure():
    tests_root = PROJECT_ROOT / TESTS_ROOT_NAME
    
    # 1. 確保根測試目錄和其 __init__.py 檔案存在
    tests_root.mkdir(exist_ok=True)
    (tests_root / "__init__.py").touch()

    # 2. 使用 itertools.chain 將多個 rglob 的結果高效地串接起來
    all_source_files = itertools.chain.from_iterable(
        (PROJECT_ROOT / src).rglob("*.py") for src in SOURCE_DIRS if (PROJECT_ROOT / src).is_dir()
    )

    # 3. 遍歷所有來源檔案，一次性完成目錄和檔案的建立
    for src_file in all_source_files:
        if src_file.name in EXCLUDE_FILES:
            continue

        # 一行程式碼完成從來源路徑到測試路徑的轉換
        relative_path = src_file.relative_to(PROJECT_ROOT)
        test_file = tests_root / relative_path.with_name(f"test_{src_file.name}")
        
        # 建立測試檔案前，確保其父目錄已存在
        test_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 在對應的測試子目錄中建立 __init__.py，將其標記為 package
        (test_file.parent / "__init__.py").touch()

        # 建立空的測試檔案
        if not test_file.exists():
            test_file.touch()
            print(f"Created: {test_file.relative_to(PROJECT_ROOT)}")

    print("\n✅ 測試目錄結構建立完成！")

if __name__ == "__main__":
    create_test_structure()