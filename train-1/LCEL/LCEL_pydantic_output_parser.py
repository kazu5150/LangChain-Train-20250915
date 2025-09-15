from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


class Recipe(BaseModel):
    ingredients: list[str] = Field(description="List of ingredients required for the recipe")
    step: list[str] = Field(description="Step-by-step instructions to prepare the recipe")


prompt = ChatPromptTemplate.from_messages([
    ("system", "ユーザーが入力した料理のレシピを考えてください\n\n{format_instructions}"),
    ("human", "{dish}"),
])

output_parser = PydanticOutputParser(pydantic_object=Recipe)

prompt_with_format_instructions = prompt.partial(
    format_instructions=output_parser.get_format_instructions()
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind(
    response_format={"type": "json_object"}
)

chain = prompt_with_format_instructions | model | output_parser
response = chain.invoke({"dish": "カレーライス"})
print(response)

