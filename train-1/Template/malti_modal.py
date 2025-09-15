from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# .env ファイルを読み込む
load_dotenv()


prompt = ChatPromptTemplate.from_messages(
    
    [
        (
            "user",
            [
                {"type": "text", "text": "画像を説明してください"},
                {"type": "image_url", "image_url": {"url":"{image_url}"}},
            ],
        )
    ]    
)

image_url = "https://pbs.twimg.com/media/GnW_apea4AAEP01?format=jpg&name=large"
# image_url = "https://cdn.sanity.io/images/9r24npb8/production/53c1c9b96a9a9560b3377b609154f5f2a9bf5da4-1200x630.jpg?auto=format&fit=max&q=75&w=1200"
prompt_value = prompt.invoke({"image_url": image_url})
print(prompt_value)


model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
ai_message = model.invoke(prompt_value.messages)
print(ai_message.content)