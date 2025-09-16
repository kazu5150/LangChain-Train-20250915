from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

cot_prompt = ChatPromptTemplate.from_template([
    ("system", " ユーザーの質問にステップバイステップで回答をしてください。"),
    ("user", "{question}"),
])

cot_chain = cot_prompt | model | StrOutputParser()

summarize_prompt = ChatPromptTemplate.from_template([
    ("system", "以下の内容を要約してください。"),
    ("user", "{content}"),
])
summarize_chain = summarize_prompt | model | StrOutputParser()


cot_summarize_chain = cot_chain | summarize_chain
cot_summarize_chain.invoke({"question": "地球温暖化の主な原因は何ですか？"})
