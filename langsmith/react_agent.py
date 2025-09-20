from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
import asyncio
from dotenv import load_dotenv
from langsmith import Client
load_dotenv()

# LangSmithクライアントの初期化とプロジェクト作成
try:
    client = Client()
    project_name = "langsmith_train"

    # プロジェクトの存在確認と作成
    try:
        client.create_project(project_name=project_name, description="ReAct Agent Training Project")
        print(f"プロジェクト '{project_name}' を作成しました。")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"プロジェクト '{project_name}' は既に存在します。")
        else:
            print(f"プロジェクト作成エラー: {e}")

except Exception as e:
    print(f"LangSmith接続エラー: {e}")
    print("LangSmithなしで実行を続行します。")

# Agent setup
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)


# Agent execution function
# 入力
inputs = [
    "2023年の時点でカナダに住んでいる人は何人ですか?",
    "デュア・リパのボーイフレンドは誰ですか?彼の年齢を.43乗すると何になりますか?",
    "デュア・リパのボーイフレンドの年齢を.43乗すると何になりますか?",
    "パリからボストンまでの距離は何マイルですか",
    "2023年のスーパーボウルで獲得された合計ポイント数は何ですか? その数値を 0.23 乗すると何になりますか?",
    "2023年のスーパーボウルで獲得された合計ポイント数を .23 乗すると何になりますか?",
    "2023年のスーパーボウルでは、2022年のスーパーボウルよりも何点多く得点されましたか?",
    "153 の 0.1312 乗は何ですか?",
    "ケンダル・ジェンナーのボーイフレンドは誰ですか? 彼の身長 (インチ) を 0.13 乗すると何ですか?",
    "1213 を 4345 で割った値は何ですか?",
]
results = []


# 非同期処理関数
async def run_agent(input_text):
    try:
        return agent.run(input_text)
    except Exception as e:
        return str(e)


# 複数Agentの同時実行
async def main():
    for input_text in inputs:
        results.append(run_agent(input_text))
    return await asyncio.gather(*results)

# 実行
if __name__ == "__main__":
    final_results = asyncio.run(main())

    # 結果の表示
    for i, (input_text, result) in enumerate(zip(inputs, final_results)):
        print(f"\n--- 質問 {i+1} ---")
        print(f"入力: {input_text}")
        print(f"結果: {result}")
        print("-" * 50)
