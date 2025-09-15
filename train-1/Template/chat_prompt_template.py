from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "以下の料理のレシピを教えてください。"),
    ("user", "料理名： {dish}"),
])
prompt_value = prompt.invoke({"dish": "カレーライス"})
print(prompt_value)
