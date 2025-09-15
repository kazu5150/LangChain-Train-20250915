from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーが入力したテーマの抒情的な詩を書いてください。"),
        ("user", ": {theme}"),
    ]
)

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


# チェーンの作成と実行 Langchain Expression Language (LCEL) を使用
# | 演算子でチェーンを接続
chain = prompt | model | StrOutputParser()

if __name__ == "__main__":
    theme = input("詩のテーマを入力してください: ")
    response = chain.invoke({"theme": theme})
    print(response)