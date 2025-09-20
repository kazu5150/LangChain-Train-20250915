from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


model = ChatOpenAI(model="gpt-4o-mini", temperature=1)
output_parsers = StrOutputParser()

cot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーの質問に対して、ステップバイステップで考えを述べてください。"),
        ("human", "{question}")
    ]
)

cot_chain = cot_prompt | model | output_parsers
# print(cot_chain.invoke({"question": "映画マイティソーの魅力を教えてください。"}))

summarize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ステップバイステップで考えた回答から結論だけ抽出してください。"),
        ("human", "{text}")
    ]
)

summarize_chain = summarize_prompt | model | output_parsers
# print(summarize_chain.invoke({"text": cot_chain.invoke({"question": "映画マイティソーの魅力を教えてください。"})}))


cot_summarize_chain = cot_chain | summarize_chain
print(cot_summarize_chain.invoke({"question": "10+20*30/15-3の計算結果を教えてください。"}))