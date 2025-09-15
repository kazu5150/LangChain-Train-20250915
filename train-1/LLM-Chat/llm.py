from langchain_openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


model = OpenAI(model="gpt-4o-mini", temperature=0)

response = model.invoke("こんにちは")
print(response)

