from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate


prompt = ChatPromptTemplate.from_messages([
    AIMessage(content="あなたは親切なアシスタントです。"),
    HumanMessage(content="{input}"),
    MessagesPlaceholder("chat_history", optional=True)
])

prompt_value = prompt.invoke({
    "input": "こんにちは！",
    "chat_history": [
        HumanMessage(content="おはようございます！"),
        AIMessage(content="おはようございます！今日はどのようなご用件でしょうか？"),
        HumanMessage(content="今日は天気がいいですね。"),
        AIMessage(content="本当にそうですね！散歩に出かけるのも良いかもしれません。")
    ]
})
print(prompt_value)
