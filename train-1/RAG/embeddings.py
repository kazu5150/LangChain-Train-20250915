from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

query = "AWSのS3からデータを読み込むためのDocument loaderはありますか？"

vector = embeddings.embed_query(query)
print(len(vector))  # ベクトルの次元数の表示
print(vector)  # 埋め込みベクトルの表示
