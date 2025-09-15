from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="今日は月曜日ですが、祝日でお休みです。langchainの勉強をしています。"),
]

for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)
