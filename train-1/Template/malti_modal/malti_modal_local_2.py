from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import base64

# .env から OPENAI_API_KEY を読み込み
load_dotenv()

# 1. 画像ファイルをBase64に変換
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"  # pngなら image/png

# ここにローカル画像のパスを指定
image_path = "image/instructions.jpeg"
image_base64 = encode_image(image_path)

prompt = """
この画像を解析して、取得できた値を構造化して出力してください。
取得したい値は以下のとおりです。
ステップバイステップで考えて情報を取得してください。

・作成者
・納期
・納品方法
・取引先名
・案件名
・予定件数
・システム作業者
・作業内容概略
"""


# 2. プロンプト作成
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_base64}},
            ],
        )
    ]
)

# 3. プロンプトをモデルに渡す
model = ChatOpenAI(model="gpt-5", temperature=0)
# model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt_value = prompt.invoke({})
ai_message = model.invoke(prompt_value.messages)

# 4. 結果を出力
print(ai_message.content)
