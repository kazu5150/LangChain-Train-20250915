"""
SerpAPI + LangSmith デモ
- Google検索の実行
- 検索結果の要約
- LangSmithでのトレース確認
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.utilities import SerpAPIWrapper
from langsmith import traceable, Client
from dotenv import load_dotenv
import os

load_dotenv()

# LangSmithの設定
client = Client()
project_name = "serpapi_demo"
os.environ["LANGCHAIN_PROJECT"] = project_name

# プロジェクト作成
try:
    client.create_project(project_name=project_name, description="SerpAPI + LangSmith Demo")
    print(f"プロジェクト '{project_name}' を作成しました。")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"プロジェクト '{project_name}' は既に存在します。")

# 初期化
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
search = SerpAPIWrapper()

@traceable(name="Google検索")
def google_search(query: str) -> str:
    """Google検索を実行"""
    try:
        results = search.run(query)
        return results
    except Exception as e:
        return f"検索エラー: {str(e)}"

@traceable(name="検索結果要約")
def summarize_search_results(query: str, search_results: str) -> str:
    """検索結果を要約"""
    messages = [
        SystemMessage(content="以下の検索結果を3-4文で要約してください。重要なポイントを簡潔にまとめてください。"),
        HumanMessage(content=f"検索クエリ: {query}\n\n検索結果:\n{search_results}")
    ]

    response = llm.invoke(messages)
    return response.content

@traceable(name="質問回答システム")
def answer_with_search(question: str) -> dict:
    """質問に対して検索を行い、回答を生成"""

    # ステップ1: Google検索
    search_results = google_search(question)

    # ステップ2: 検索結果の要約
    summary = summarize_search_results(question, search_results)

    # ステップ3: 最終回答の生成
    messages = [
        SystemMessage(content="検索結果を基に、ユーザーの質問に対して正確で分かりやすい回答をしてください。"),
        HumanMessage(content=f"質問: {question}\n\n検索結果要約: {summary}")
    ]

    final_answer = llm.invoke(messages)

    return {
        "question": question,
        "search_results": search_results[:500] + "..." if len(search_results) > 500 else search_results,
        "summary": summary,
        "final_answer": final_answer.content
    }

@traceable(name="最新ニュース検索")
def get_latest_news(topic: str) -> str:
    """特定トピックの最新ニュースを検索"""
    news_query = f"{topic} 最新ニュース 2024"
    results = google_search(news_query)

    messages = [
        SystemMessage(content="以下のニュース検索結果から、最新の重要なニュースを3つピックアップして要約してください。"),
        HumanMessage(content=f"トピック: {topic}\n\n検索結果:\n{results}")
    ]

    response = llm.invoke(messages)
    return response.content

def main():
    print("=== SerpAPI + LangSmith デモ ===")
    print("Google検索とLLMを組み合わせた質問応答システム\n")

    # テスト質問
    test_questions = [
        "2025年の最新AIエージェントツールは？",
        "アレックス　フーバーというクライマーについて教えてください。",
        "クライミングWC 2025年の森秋彩選手の結果を教えてください。"
    ]

    # ニュース検索テスト
    news_topics = [
        "クライミング",
        "モーグル"
    ]

    print("=== 質問応答テスト ===")
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 質問 {i} ---")
        print(f"質問: {question}")

        try:
            result = answer_with_search(question)
            print(f"最終回答: {result['final_answer']}")
            print("-" * 70)
        except Exception as e:
            print(f"エラー: {e}")

    print("\n=== ニュース検索テスト ===")
    for i, topic in enumerate(news_topics, 1):
        print(f"\n--- ニューストピック {i} ---")
        print(f"トピック: {topic}")

        try:
            news = get_latest_news(topic)
            print(f"最新ニュース: {news}")
            print("-" * 70)
        except Exception as e:
            print(f"エラー: {e}")

    print(f"\n✅ SerpAPIデモ完了！")
    print(f"LangSmithプロジェクト '{project_name}' で以下を確認してください：")
    print("- Google検索のトレース")
    print("- 検索結果要約のトレース")
    print("- 質問応答システムの処理フロー")
    print("- 各ステップの入力・出力データ")

if __name__ == "__main__":
    main()