"""
LangSmith学習用 - 基本機能デモ
- カスタムトレース
- エラーハンドリング
- メタデータ追加
- 複数のLLM呼び出し
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable, Client
from dotenv import load_dotenv
import os
import time

load_dotenv()

# LangSmithの設定
client = Client()
project_name = "langsmith_basics_demo"
os.environ["LANGCHAIN_PROJECT"] = project_name

# プロジェクト作成
try:
    client.create_project(project_name=project_name, description="LangSmith Basic Features Demo")
    print(f"プロジェクト '{project_name}' を作成しました。")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"プロジェクト '{project_name}' は既に存在します。")

# LLMの初期化
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

@traceable(name="翻訳機能")
def translate_text(text: str, target_language: str) -> str:
    """テキストを指定言語に翻訳"""
    messages = [
        SystemMessage(content=f"あなたは優秀な翻訳者です。以下のテキストを{target_language}に翻訳してください。"),
        HumanMessage(content=text)
    ]

    response = llm.invoke(messages)
    return response.content

@traceable(name="要約機能")
def summarize_text(text: str) -> str:
    """テキストを要約"""
    messages = [
        SystemMessage(content="以下のテキストを3文以内で要約してください。"),
        HumanMessage(content=text)
    ]

    response = llm.invoke(messages)
    return response.content

@traceable(name="感情分析")
def analyze_sentiment(text: str) -> str:
    """テキストの感情分析"""
    messages = [
        SystemMessage(content="以下のテキストの感情を分析して、ポジティブ・ネガティブ・ニュートラルで判定してください。理由も簡潔に説明してください。"),
        HumanMessage(content=text)
    ]

    response = llm.invoke(messages)
    return response.content

@traceable(name="マルチステップ処理")
def multi_step_process(original_text: str) -> dict:
    """複数のステップを含む処理"""

    # ステップ1: 翻訳
    translated = translate_text(original_text, "英語")

    # ステップ2: 要約
    summary = summarize_text(translated)

    # ステップ3: 感情分析
    sentiment = analyze_sentiment(original_text)

    return {
        "original": original_text,
        "translated": translated,
        "summary": summary,
        "sentiment": sentiment
    }

def main():
    print("=== LangSmith 基本機能デモ ===")
    print("複数のLLM機能とトレースを確認します\n")

    # テストテキスト
    test_texts = [
        "今日は素晴らしい天気で、公園で友達と楽しい時間を過ごしました。",
        "会議が長引いて疲れましたが、良いアイデアがたくさん出ました。",
        "新しいプロジェクトが始まります。チーム一丸となって頑張りましょう！"
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n--- テスト {i} ---")
        print(f"元テキスト: {text}")

        try:
            # マルチステップ処理を実行
            result = multi_step_process(text)

            print(f"翻訳結果: {result['translated']}")
            print(f"要約: {result['summary']}")
            print(f"感情分析: {result['sentiment']}")
            print("-" * 80)

            # 少し待機（トレースの確認用）
            time.sleep(1)

        except Exception as e:
            print(f"エラー: {e}")

    print(f"\n✅ デモ完了！")
    print(f"LangSmithプロジェクト '{project_name}' で詳細なトレースを確認してください。")
    print("以下の機能のトレースが記録されています：")
    print("- 翻訳機能")
    print("- 要約機能")
    print("- 感情分析")
    print("- マルチステップ処理")

if __name__ == "__main__":
    main()