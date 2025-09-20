"""
LangSmith基本学習用 - 超シンプルなチャットボット
LangSmithでトレースを確認するための最小限のプログラム
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

# LangSmithプロジェクト名を設定
project_name = "simple_chat_demo"
os.environ["LANGCHAIN_PROJECT"] = project_name

print(f"LangSmithプロジェクト: {project_name}")

# ChatOpenAIモデルの初期化
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)


def simple_chat(user_input: str) -> str:
    """シンプルなチャット機能"""
    messages = [
        SystemMessage(content="あなたは親切な日本語アシスタントです。"),
        HumanMessage(content=user_input)
    ]

    response = llm.invoke(messages)
    return response.content


def main():
    print("=== 超シンプル チャットボット ===")
    print("LangSmithでトレースを確認しよう！\n")

    # テスト用の質問
    test_questions = [
        "こんにちは！",
        "Pythonって何ですか？",
        "LangChainについて教えて",
        "ありがとう！"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"--- 質問 {i} ---")
        print(f"ユーザー: {question}")

        answer = simple_chat(question)
        print(f"アシスタント: {answer}")
        print("-" * 40)

    print("\n✅ 完了！")
    print("LangSmithで実行履歴を確認してみよう: https://smith.langchain.com")


if __name__ == "__main__":
    main()