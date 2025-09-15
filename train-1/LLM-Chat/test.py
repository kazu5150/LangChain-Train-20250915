from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

message = [
    SystemMessage(content="あなたは翻訳するプロです。"),
    HumanMessage(content="「こんにちは、元気ですか？」をタイ語にを翻訳してください。")
]

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

response = llm.invoke(message)
print(response.content)
