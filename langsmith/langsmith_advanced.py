"""
LangSmith上級学習用 - エラーハンドリングとメタデータ
- カスタムメタデータ
- エラートレース
- パフォーマンス計測
- 条件分岐のトレース
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable, Client
from dotenv import load_dotenv
import os
import time
import random

load_dotenv()

# LangSmithの設定
client = Client()
project_name = "langsmith_advanced_demo"
os.environ["LANGCHAIN_PROJECT"] = project_name

# プロジェクト作成
try:
    client.create_project(project_name=project_name, description="LangSmith Advanced Features - Error Handling & Metadata")
    print(f"プロジェクト '{project_name}' を作成しました。")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"プロジェクト '{project_name}' は既に存在します。")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

@traceable(name="タスク分類器")
def classify_task(user_input: str) -> str:
    """ユーザー入力のタスクを分類"""
    messages = [
        SystemMessage(content="ユーザーの入力を以下のカテゴリーに分類してください：質問、依頼、挨拶、その他。カテゴリー名のみ回答してください。"),
        HumanMessage(content=user_input)
    ]

    response = llm.invoke(messages)
    return response.content.strip()

@traceable(name="質問回答システム")
def answer_question(question: str) -> str:
    """質問に回答"""
    messages = [
        SystemMessage(content="質問に対して正確で簡潔な回答をしてください。"),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages)
    return response.content

@traceable(name="依頼処理システム")
def handle_request(request: str) -> str:
    """依頼を処理"""
    messages = [
        SystemMessage(content="依頼に対して具体的な手順や方法を提示してください。"),
        HumanMessage(content=request)
    ]

    response = llm.invoke(messages)
    return response.content

@traceable(name="挨拶応答システム")
def handle_greeting(greeting: str) -> str:
    """挨拶に応答"""
    responses = [
        "こんにちは！今日も良い一日ですね。",
        "お疲れ様です！何かお手伝いできることはありますか？",
        "こんにちは！元気にしていますか？",
        "いらっしゃいませ！今日はどのようなご用件でしょうか？"
    ]
    return random.choice(responses)

@traceable(name="エラーシミュレータ")
def simulate_error(text: str) -> str:
    """意図的にエラーを発生させる（テスト用）"""
    if "エラー" in text:
        raise ValueError("意図的なエラー: 'エラー'という文字が含まれています")

    return f"正常処理: {text}"

@traceable(
    name="スマートアシスタント",
    metadata={"version": "1.0", "feature": "intelligent_routing"}
)
def smart_assistant(user_input: str) -> dict:
    """スマートアシスタントのメイン処理"""

    start_time = time.time()

    try:
        # ステップ1: タスク分類
        task_type = classify_task(user_input)

        # ステップ2: エラーシミュレーション
        simulate_error(user_input)

        # ステップ3: タスクタイプに応じた処理
        if "質問" in task_type:
            response = answer_question(user_input)
            handler = "question_handler"
        elif "依頼" in task_type:
            response = handle_request(user_input)
            handler = "request_handler"
        elif "挨拶" in task_type:
            response = handle_greeting(user_input)
            handler = "greeting_handler"
        else:
            response = "申し訳ございませんが、理解できませんでした。"
            handler = "default_handler"

        processing_time = time.time() - start_time

        return {
            "input": user_input,
            "task_type": task_type,
            "response": response,
            "handler": handler,
            "processing_time": processing_time,
            "status": "success"
        }

    except Exception as e:
        processing_time = time.time() - start_time

        return {
            "input": user_input,
            "error": str(e),
            "processing_time": processing_time,
            "status": "error"
        }

def main():
    print("=== LangSmith 上級機能デモ ===")
    print("エラーハンドリング、メタデータ、条件分岐のトレースを確認します\n")

    # テスト入力（正常パターンとエラーパターン）
    test_inputs = [
        "こんにちは！",
        "Pythonの勉強方法を教えてください",
        "レポートを書くのを手伝ってください",
        "AIとは何ですか？",
        "この文章にはエラーという単語が含まれています",  # エラーを意図的に発生
        "今日の天気はどうですか？",
        "ありがとうございました"
    ]

    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n--- テスト {i} ---")
        print(f"入力: {user_input}")

        result = smart_assistant(user_input)

        if result["status"] == "success":
            print(f"タスクタイプ: {result['task_type']}")
            print(f"ハンドラー: {result['handler']}")
            print(f"応答: {result['response']}")
            print(f"処理時間: {result['processing_time']:.2f}秒")
        else:
            print(f"エラー: {result['error']}")
            print(f"処理時間: {result['processing_time']:.2f}秒")

        print("-" * 80)
        time.sleep(0.5)

    print(f"\n✅ 上級デモ完了！")
    print(f"LangSmithプロジェクト '{project_name}' で以下を確認してください：")
    print("- 正常処理のトレース")
    print("- エラー発生時のトレース")
    print("- 条件分岐の処理フロー")
    print("- 処理時間などのメタデータ")
    print("- カスタム関数のトレース階層")

if __name__ == "__main__":
    main()