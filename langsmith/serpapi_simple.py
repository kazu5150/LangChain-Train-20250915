"""
SerpAPI 基本学習用 - シンプルな検索デモ
LangSmithでSerpAPIの使用をトレースする最小限のプログラム
"""

from langchain_community.utilities import SerpAPIWrapper
from langsmith import traceable, Client
from dotenv import load_dotenv
import os

load_dotenv()

# LangSmithの設定
client = Client()
project_name = "serpapi_simple"
os.environ["LANGCHAIN_PROJECT"] = project_name

# プロジェクト作成
try:
    client.create_project(project_name=project_name, description="Simple SerpAPI Demo")
    print(f"プロジェクト '{project_name}' を作成しました。")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"プロジェクト '{project_name}' は既に存在します。")

# SerpAPI初期化
search = SerpAPIWrapper()

@traceable(name="シンプル検索")
def simple_search(query: str) -> str:
    """シンプルなGoogle検索"""
    try:
        results = search.run(query)
        return results
    except Exception as e:
        return f"検索エラー: {str(e)}"

@traceable(name="検索結果フィルタ")
def filter_results(results: str, max_length: int = 300) -> str:
    """検索結果を指定文字数でフィルタ"""
    if len(results) <= max_length:
        return results

    filtered = results[:max_length] + "..."
    return filtered

@traceable(name="複数キーワード検索")
def multi_keyword_search(keywords: list) -> dict:
    """複数のキーワードで検索を実行"""
    results = {}

    for keyword in keywords:
        search_result = simple_search(keyword)
        filtered_result = filter_results(search_result)
        results[keyword] = filtered_result

    return results

def main():
    print("=== SerpAPI シンプルデモ ===")
    print("基本的なGoogle検索機能のテスト\n")

    # 単一検索テスト
    single_queries = [
        "Python programming",
        "LangChain tutorial",
        "OpenAI GPT-4"
    ]

    print("=== 単一検索テスト ===")
    for i, query in enumerate(single_queries, 1):
        print(f"\n--- 検索 {i} ---")
        print(f"クエリ: {query}")

        try:
            result = simple_search(query)
            filtered = filter_results(result)
            print(f"結果: {filtered}")
            print("-" * 50)
        except Exception as e:
            print(f"エラー: {e}")

    # 複数キーワード検索テスト
    keywords = ["AI", "機械学習", "深層学習"]

    print(f"\n=== 複数キーワード検索テスト ===")
    print(f"キーワード: {keywords}")

    try:
        multi_results = multi_keyword_search(keywords)

        for keyword, result in multi_results.items():
            print(f"\n[{keyword}] の検索結果:")
            print(result[:200] + "..." if len(result) > 200 else result)
            print("-" * 30)

    except Exception as e:
        print(f"エラー: {e}")

    print(f"\n✅ シンプルデモ完了！")
    print(f"LangSmithプロジェクト '{project_name}' で確認してください：")
    print("- SerpAPIの基本的な使用方法")
    print("- 検索結果のフィルタリング")
    print("- 複数キーワードでの連続検索")
    print("- 各検索のトレース記録")

if __name__ == "__main__":
    main()