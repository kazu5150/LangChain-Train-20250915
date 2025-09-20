"""
LangSmith基本学習用 - シンプルなチャットボット
LangSmithでトレースを確認するための最小限のプログラム
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langsmith import Client
import os

load_dotenv()

# LangSmithクライアントの初期化
try:
    client = Client()
    project_name = "simple_chat_demo"

    # プロジェクトの存在確認と作成
    try:
        client.create_project(project_name=project_name, description="Simple Chat Demo for LangSmith Learning")
        print(f"プロジェクト '{project_name}' を作成しました。")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"プロジェクト '{project_name}' は既に存在します。")
        else:
            print(f"プロジェクト作成エラー: {e}")

except Exception as e:
    print(f"LangSmith接続エラー: {e}")

# LangSmithプロジェクト環境変数を設定
os.environ["LANGCHAIN_PROJECT"] = project_name

# ChatOpenAIモデルの初期化
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=200
)

def simple_chat(user_input: str) -> str:
    """シンプルなチャット機能"""
    messages = [
        SystemMessage(content="あなたは親切で丁寧な日本語アシスタントです。簡潔に回答してください。"),
        HumanMessage(content=user_input)
    ]

    response = llm.invoke(messages)
    return response.content

def main():
    print("=== LangSmith Simple Chat Demo ===")
    print("LangSmithでトレースを確認できる簡単なチャットボットです")
    print("'quit'で終了\n")

    # テスト用の質問リスト
    test_questions = [
        "こんにちは！今日はいい天気ですね。",
        "Pythonについて教えてください。",
        "LangChainとは何ですか？",
        "おすすめの勉強方法を教えてください。",
        "ありがとうございました！"
    ]

    print("自動テスト実行中...")
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 質問 {i} ---")
        print(f"ユーザー: {question}")

        try:
            answer = simple_chat(question)
            print(f"アシスタント: {answer}")
            print("-" * 50)
        except Exception as e:
            print(f"エラー: {e}")

    print(f"\n✅ テスト完了！LangSmithプロジェクト '{project_name}' でトレースを確認してください。")
    print("https://smith.langchain.com でプロジェクトを開いて実行履歴を見ることができます。")

if __name__ == "__main__":
    main()