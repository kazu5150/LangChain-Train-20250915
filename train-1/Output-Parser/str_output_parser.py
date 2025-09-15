from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
template = ChatPromptTemplate.from_messages([
    SystemMessage(content="あなたは日本語を映画のタイトルのようなに表現豊かに表現するアシスタントです。サブタイトルも付け加えます。"),
    HumanMessage(content="夏への扉"),
])
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
parser = StrOutputParser()

chain = template | llm | parser
result = chain.invoke({})
print(result)
