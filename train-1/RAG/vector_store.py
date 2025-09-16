# P94
# RAGシステムの実装：LangChainのGitリポジトリから文書を読み込み、ベクターデータベースを作成して検索を行う

from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from dotenv import load_dotenv

# 環境変数を読み込み（OpenAI APIキーなど）
load_dotenv()


def file_filter(file_path: str) -> bool:
    """
    ファイルフィルター関数：.mdまたは.mdxファイルを対象とする

    Args:
        file_path (str): ファイルパス

    Returns:
        bool: .mdまたは.mdxファイルの場合True、それ以外はFalse
    """
    return file_path.endswith(('.md', '.mdx'))


# GitLoaderを使用してLangChainリポジトリから文書を読み込み
loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

# 生の文書を読み込み
raw_docs = loader.load()
print(f"読み込み文書数: {len(raw_docs)}")

# 取得したファイルの拡張子を確認
print("\n=== 取得したファイルの拡張子確認 ===")
file_extensions = {}
for doc in raw_docs[:10]:  # 最初の10件をチェック
    file_path = doc.metadata.get('file_path', '')
    if file_path:
        ext = file_path.split('.')[-1] if '.' in file_path else 'no_extension'
        file_extensions[ext] = file_extensions.get(ext, 0) + 1
        print(f"ファイルパス: {file_path}")

print(f"\n拡張子の分布（最初の10件）: {file_extensions}")

print("\n=== 生文書の最初のサンプル ===")
if raw_docs:
    first_raw_doc = raw_docs[0]
    print(f"メタデータ: {first_raw_doc.metadata}")
    print(f"内容（最初の500文字）: {first_raw_doc.page_content[:500]}...")
    print(f"文書の全文字数: {len(first_raw_doc.page_content)}")

# テキストスプリッターで文書をチャンクに分割
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(raw_docs)
print(f"\n分割後のチャンク数: {len(docs)}")
print("\n=== 分割後チャンクの最初のサンプル ===")
if docs:
    first_chunk = docs[0]
    print(f"チャンクのメタデータ: {first_chunk.metadata}")
    print(f"チャンクの内容: {first_chunk.page_content[:300]}...")
    print(f"チャンクの文字数: {len(first_chunk.page_content)}")

# OpenAIのembeddingモデルを使用してベクター化
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Chromaベクターデータベースを作成
print(f"\n=== ベクターデータベース作成中 ===")
db = Chroma.from_documents(docs, embeddings)
print(f"ベクターデータベースの作成完了")

# ベクター化のサンプルを確認（最初のチャンクをベクター化して確認）
print("\n=== ベクターデータのサンプル ===")
if docs:
    sample_text = docs[0].page_content[:200]  # 最初の200文字
    sample_vector = embeddings.embed_query(sample_text)
    print(f"サンプルテキスト: {sample_text}...")
    print(f"ベクターの次元数: {len(sample_vector)}")
    print(f"ベクターの最初の10要素: {sample_vector[:10]}")

# 検索用のretrieverを作成
retriever = db.as_retriever()

# 検索クエリ
query = "AWSのS3からデータを読み込むためのDocument loaderはありますか？"

# 関連文書を検索
print(f"\n=== 検索実行 ===")
print(f"検索クエリ: {query}")
context_docs = retriever.invoke(query)
print(f"検索結果数: {len(context_docs)}")

# 検索結果を詳細表示
print(f"\n=== 検索結果の詳細 ===")
for i, doc in enumerate(context_docs):
    print(f"\n--- 検索結果 {i+1} ---")
    print(f"メタデータ: {doc.metadata}")
    print(f"内容（最初の300文字）: {doc.page_content[:300]}...")
    print(f"文書の文字数: {len(doc.page_content)}")
    if i >= 2:  # 最初の3件のみ表示
        print(f"\n（残り{len(context_docs)-3}件の結果は省略）")
        break
