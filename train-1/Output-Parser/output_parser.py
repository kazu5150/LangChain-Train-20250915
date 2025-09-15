# output_parser_demo_jp.py
# ------------------------------------------------------------
# 必要: pip install langchain-core langchain-openai pydantic
# 環境変数: export OPENAI_API_KEY=sk-...
# ------------------------------------------------------------

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # .envから環境変数をロード


# 1) Pydanticモデル定義
class Recipe(BaseModel):
    ingredients: list[str] = Field(description="料理に使う材料")
    steps: list[str] = Field(description="料理を作る手順")


# 2) PydanticOutputParserを作成
parser = PydanticOutputParser(pydantic_object=Recipe)

# 3) LLMに渡す「出力フォーマット指示」を取得
format_instructions = parser.get_format_instructions()

# 4) 日本語のプロンプトを作成
prompt = ChatPromptTemplate.from_template(
    "あなたは料理アシスタントです。\n"
    "以下の料理名に対して、シンプルなレシピを考えてください。\n"
    "必ず次のフォーマットに従って、**JSONのみ**を出力してください。\n\n"
    "{format_instructions}\n"
    "重要:\n"
    "- JSON以外の説明文やコードブロックは一切出力しないでください。\n"
    "- 材料は文字列の配列、手順も文字列の配列で書いてください。\n\n"
    "料理名: {dish}"
)

# 5) モデルを初期化
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 6) プロンプト → モデル → パーサ のチェーンを作成
chain = prompt | llm | parser


if __name__ == "__main__":
    # 7) 実行（お好きな料理名をどうぞ）
    dish = "パエリア"
    result: Recipe = chain.invoke({
        "dish": dish,
        "format_instructions": format_instructions
    })

    # 8) Pydanticモデルとして安全に扱える
    print("== 解析結果 ==")
    print(result.model_dump_json(indent=2))

    print("\n== 材料 ==")
    for i, ing in enumerate(result.ingredients, 1):
        print(f"{i}. {ing}")

    print("\n== 手順 ==")
    for i, step in enumerate(result.steps, 1):
        print(f"{i}. {step}")
