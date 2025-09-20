from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()


def calculate_word_similarity():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    word1 = input("最初の単語を入力してください: ")
    word2 = input("二番目の単語を入力してください: ")

    embed1 = embeddings.embed_query(word1)
    embed2 = embeddings.embed_query(word2)

    embed1 = np.array(embed1)
    embed2 = np.array(embed2)

    dot_product = np.dot(embed1, embed2)
    norm_a = np.linalg.norm(embed1)
    norm_b = np.linalg.norm(embed2)
    similarity = dot_product / (norm_a * norm_b)

    print(f"'{word1}' と '{word2}' の類似度: {similarity:.4f}")

    if similarity >= 0.9:
        interpretation = "非常に高い類似度です。ほぼ同義語レベルです。"
    elif similarity >= 0.7:
        interpretation = "高い類似度です。関連性が強い単語です。"
    elif similarity >= 0.5:
        interpretation = "中程度の類似度です。ある程度の関連性があります。"
    elif similarity >= 0.3:
        interpretation = "低い類似度です。関連性は弱いです。"
    else:
        interpretation = "非常に低い類似度です。ほとんど関連性がありません。"

    print(f"評価: {interpretation}")

    return similarity


if __name__ == "__main__":
    calculate_word_similarity()