from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage 
from dotenv import load_dotenv

# .env ファイルを読み込む
load_dotenv()

# ✅ api_key は渡さず、環境変数に任せる
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="こんにちは、私はKazuと言います。"),
    AIMessage(content="こんにちは、Kazuさん！今日はどのようにお手伝いできますか？"),
    HumanMessage(content="私の名前がわかりますか？"),
]

ai_message = model.invoke(messages)
print(ai_message.content)
