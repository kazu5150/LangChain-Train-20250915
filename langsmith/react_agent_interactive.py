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
    project_name = "langsmith_train_interactive"

    # プロジェクトの存在確認と作成
    try:
        client.create_project(
            project_name=project_name,
            description="ReAct Agent Interactive Training Project"
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

# Agent setup
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)


def show_available_tools():
    """利用可能なツールを表示する"""
    print("🔧 利用可能なツール:")
    for i, tool in enumerate(tools, 1):
        print(f"  {i}. {tool.name}: {tool.description}")
    print()


def run_agent_interactive():
    """ユーザーから質問を入力として受け取り、エージェントを実行する"""
    print("=== ReAct Agent Interactive Mode ===")
    print("このエージェントは以下のツールを使用できます：\n")

    show_available_tools()

    print("質問を入力してください（'quit'で終了、'tools'でツール一覧表示）")
    print("例: '2023年の時点でカナダに住んでいる人は何人ですか?'")
    print("-" * 50)

    while True:
        user_input = input("\n質問を入力してください: ").strip()

        if user_input.lower() in ['quit', 'exit', '終了', 'q']:
            print("プログラムを終了します。")
            break

        if user_input.lower() in ['tools', 'ツール']:
            show_available_tools()
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

        print("-" * 50)


# 実行
if __name__ == "__main__":
    run_agent_interactive()