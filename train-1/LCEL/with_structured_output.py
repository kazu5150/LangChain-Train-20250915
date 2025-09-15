from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()


class Recipe(BaseModel):
    ingredients: list[str] = Field(description="List of ingredients required for the recipe")
    step: list[str] = Field(description="Step-by-step instructions to prepare the recipe")


prompt = ChatPromptTemplate.from_messages([
    ("system", "ユーザーが入力した料理のレシピを考えてください。"),
    ("human", "{dish}"),
])

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = prompt | model.with_structured_output(Recipe)

recipe = chain.invoke({"dish": "カレーライス"})
print(type(recipe))
print(recipe)
