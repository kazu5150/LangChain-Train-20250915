from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from dotenv import load_dotenv
from langsmith import Client
import os

load_dotenv()

# LangSmithクライアントの初期化とプロジェクト作成
try:
    client = Client()
    project_name = "langsmith_enhanced_agent"

    # プロジェクトの存在確認と作成
    try:
        client.create_project(
            project_name=project_name,
            description="Enhanced ReAct Agent with Multiple Tools"
        )
        print(f"プロジェクト '{project_name}' を作成しました。")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"プロジェクト '{project_name}' は既に存在します。")
        else:
            print(f"プロジェクト作成エラー: {e}")

except Exception as e:
    print(f"LangSmith接続エラー: {e}")
    print("LangSmithなしで実行を続行します。")

# LangSmithプロジェクト環境変数を設定
os.environ["LANGCHAIN_PROJECT"] = project_name

# Agent setup with enhanced tools
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
tools = load_tools(["serpapi", "llm-math", "wikipedia"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)


def show_available_tools():
    """利用可能なツールを表示する"""
    print("🔧 利用可能なツール:")

    # ツールの日本語説明
    tool_descriptions = {
        "Search": "検索エンジン - インターネット検索でリアルタイムな情報を取得",
        "Calculator": "計算機 - 数学的な計算や複雑な数式を実行",
        "Wikipedia": "百科事典 - Wikipedia検索で詳細な情報を取得"
    }

    for i, tool in enumerate(tools, 1):
        jp_description = tool_descriptions.get(tool.name, tool.description)
        print(f"  {i}. {tool.name}: {jp_description}")
    print()


def show_usage_examples():
    """使用例を表示する"""
    print("💡 使用例:")
    print("  🌐 最新情報: '2024年のノーベル物理学賞は誰が受賞した？'")
    print("  📚 百科事典: 'アインシュタインについてWikipediaで調べて'")
    print("  🧮 複雑計算: '√(2^10 + 3^5) × π の値は？'")
    print("  🔍 比較検索: '日本とドイツの人口を比較して'")
    print("  📖 詳細情報: '量子コンピュータについて詳しく教えて'")
    print()


def run_agent_interactive():
    """ユーザーから質問を入力として受け取り、エージェントを実行する"""
    print("=== Enhanced ReAct Agent Interactive Mode ===")
    print("このエージェントは以下のツールを使用できます：\n")

    show_available_tools()
    show_usage_examples()

    print("質問を入力してください")
    print("コマンド: 'quit'で終了、'tools'でツール一覧、'examples'で使用例表示")
    print("-" * 60)

    while True:
        user_input = input("\n質問を入力してください: ").strip()

        if user_input.lower() in ['quit', 'exit', '終了', 'q']:
            print("プログラムを終了します。")
            break

        if user_input.lower() in ['tools', 'ツール']:
            show_available_tools()
            continue

        if user_input.lower() in ['examples', '例', '使用例']:
            show_usage_examples()
            continue

        if not user_input:
            print("質問を入力してください。")
            continue

        print(f"\n質問: {user_input}")
        print("処理中...")

        try:
            result = agent.run(user_input)
            print(f"\n回答: {result}")
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")

        print("-" * 60)


# 実行
if __name__ == "__main__":
    run_agent_interactive()