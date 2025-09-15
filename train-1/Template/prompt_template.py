from langchain_core.prompts import PromptTemplate


prompt = PromptTemplate.from_template("""
以下の料理のレシピを教えてください。

料理名： {dish}

""")
prompt_value = prompt.invoke({"dish": "カレーライス"})
print(prompt_value)
