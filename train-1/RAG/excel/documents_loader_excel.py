from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import pandas as pd


load_dotenv()

file_path = 'data/sample.xlsx'

# Load Excel file using pandas
df = pd.read_excel(file_path)

# Convert DataFrame to documents
docs = []
for index, row in df.iterrows():
    # Convert each row to a string representation, handling NaN values
    content = ' | '.join([f"{col}: {str(row[col]) if pd.notna(row[col]) else 'N/A'}" for col in df.columns])
    doc = Document(
        page_content=content,
        metadata={"row": index, "source": file_path}
    )
    docs.append(doc)

vector_stores = Chroma.from_documents(docs, OpenAIEmbeddings())
retriever = vector_stores.as_retriever(search_kwargs={'k': 1})

# create chatbot
template = """前の会話履歴を考慮して、以下のコンテキストのみに基づいて質問に答えてください:

会話履歴:
{chat_history}

コンテキスト:
{context}

質問: {question}

答え:"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
output_parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_chat_history(chat_history):
    formatted = []
    for human, ai in chat_history:
        formatted.append(f"Human: {human}")
        formatted.append(f"AI: {ai}")
    return "\n".join(formatted)

qa_chain = (
    {
        "context": lambda x: format_docs(retriever.invoke(x["question"])),
        "question": lambda x: x["question"],
        "chat_history": lambda x: format_chat_history(x.get("chat_history", []))
    }
    | prompt
    | model
    | output_parser
)

# interaction
chat_history = []
for _ in range(10):
    query = input("ボットへの質問: ")

    # Get answer
    result = qa_chain.invoke({"question": query, "chat_history": chat_history})

    # Get source documents
    source_docs = retriever.invoke(query)

    chat_history.append((query, result))
    print("\n")
    print("Answer: ", result)
    print("\n")
    if source_docs:
        print("Source document: ", source_docs[0].page_content)
    print("\n")
