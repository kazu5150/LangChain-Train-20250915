from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, Tool
from langchain_community.agent_toolkits.load_tools import load_tools
from dotenv import load_dotenv
from langsmith import Client
import os
import numpy as np
import matplotlib.pyplot as plt

load_dotenv()

# LangSmithクライアントの初期化とプロジェクト作成
try:
    client = Client()
    project_name = "langsmith_graph_agent"

    # プロジェクトの存在確認と作成
    try:
        client.create_project(
            project_name=project_name,
            description="ReAct Agent with Graph Creation Capability"
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


def create_graph(graph_type: str) -> str:
    """グラフを作成する関数"""
    try:
        plt.figure(figsize=(10, 6))

        if "sin" in graph_type.lower() or "cos" in graph_type.lower():
            x = np.linspace(0, 4*np.pi, 100)
            if "sin" in graph_type.lower():
                y = np.sin(x)
                plt.plot(x, y, 'b-', label='sin(x)', linewidth=2)
            if "cos" in graph_type.lower():
                y = np.cos(x)
                plt.plot(x, y, 'r-', label='cos(x)', linewidth=2)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('三角関数のグラフ')
            plt.grid(True)
            plt.legend()

        elif "parabola" in graph_type.lower() or "二次" in graph_type.lower():
            x = np.linspace(-5, 5, 100)
            y = x**2
            plt.plot(x, y, 'g-', linewidth=2)
            plt.xlabel('x')
            plt.ylabel('y = x²')
            plt.title('二次関数のグラフ')
            plt.grid(True)

        elif "exponential" in graph_type.lower() or "指数" in graph_type.lower():
            x = np.linspace(-2, 3, 100)
            y = np.exp(x)
            plt.plot(x, y, 'purple', linewidth=2)
            plt.xlabel('x')
            plt.ylabel('y = e^x')
            plt.title('指数関数のグラフ')
            plt.grid(True)

        else:
            # デフォルト: sin波
            x = np.linspace(0, 2*np.pi, 100)
            y = np.sin(x)
            plt.plot(x, y, 'b-', linewidth=2)
            plt.xlabel('x')
            plt.ylabel('sin(x)')
            plt.title('sin波のグラフ')
            plt.grid(True)

        # グラフを保存
        filename = f"graph_{hash(graph_type) % 10000}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        return f"グラフを作成しました: {filename}\nグラフタイプ: {graph_type}"

    except Exception as e:
        return f"グラフ作成でエラーが発生しました: {e}"


def calculate_stats(data_range: str) -> str:
    """統計計算を行う関数"""
    try:
        if "1-100" in data_range or "1から100" in data_range:
            data = np.arange(1, 101)
        elif "1-50" in data_range or "1から50" in data_range:
            data = np.arange(1, 51)
        else:
            data = np.arange(1, 101)  # デフォルト

        mean = np.mean(data)
        std = np.std(data)
        median = np.median(data)

        return f"""統計結果:
平均値: {mean}
標準偏差: {std:.2f}
中央値: {median}
最小値: {np.min(data)}
最大値: {np.max(data)}
データ数: {len(data)}"""

    except Exception as e:
        return f"統計計算でエラーが発生しました: {e}"


# カスタムツールを作成
graph_tool = Tool(
    name="GraphCreator",
    description="グラフ作成ツール。sin, cos, parabola, exponentialなどのグラフを作成できます。使用例: 'sin cos'で三角関数を比較",
    func=create_graph
)

stats_tool = Tool(
    name="StatisticsCalculator",
    description="統計計算ツール。数値範囲の平均、標準偏差、中央値などを計算します。例: '1-100'で1から100の統計",
    func=calculate_stats
)

# Agent setup
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
base_tools = load_tools(["serpapi", "llm-math", "wikipedia"], llm=llm)
custom_tools = [graph_tool, stats_tool]
tools = base_tools + custom_tools

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
        "Search": "🔍 検索エンジン - インターネット検索でリアルタイムな情報を取得",
        "Calculator": "🧮 計算機 - 数学的な計算や複雑な数式を実行",
        "Wikipedia": "📚 百科事典 - Wikipedia検索で詳細な情報を取得",
        "GraphCreator": "📊 グラフ作成 - sin, cos, 二次関数, 指数関数のグラフを作成",
        "StatisticsCalculator": "📈 統計計算 - 数値データの平均、標準偏差、中央値を計算"
    }

    for i, tool in enumerate(tools, 1):
        jp_description = tool_descriptions.get(tool.name, tool.description)
        print(f"  {i}. {tool.name}: {jp_description}")
    print()


def show_usage_examples():
    """使用例を表示する"""
    print("💡 使用例:")
    print("  📊 グラフ作成: 'sin波とcos波のグラフを作成して'")
    print("  📈 統計計算: '1から100までの数の統計を計算して'")
    print("  🌐 最新情報: '2024年のノーベル物理学賞は誰が受賞した？'")
    print("  📚 百科事典: 'アインシュタインについてWikipediaで調べて'")
    print("  🧮 複雑計算: '√(2^10 + 3^5) × π の値は？'")
    print("  📊 関数グラフ: '二次関数のグラフを描いて'")
    print()


def run_agent_interactive():
    """ユーザーから質問を入力として受け取り、エージェントを実行する"""
    print("=== Graph-Enabled ReAct Agent Interactive Mode ===")
    print("このエージェントは検索・計算・Wikipedia・グラフ作成・統計計算ができます！\n")

    show_available_tools()
    show_usage_examples()

    print("質問を入力してください")
    print("コマンド: 'quit'で終了、'tools'でツール一覧、'examples'で使用例表示")
    print("-" * 70)

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

        print("-" * 70)


# 実行
if __name__ == "__main__":
    run_agent_interactive()