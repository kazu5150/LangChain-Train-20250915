import operator
import re
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.runnables import ConfigurableField
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# ロール定義
# ---------------------------
ROLES = {
    "1": {
        "name": "一般知識エキスパート",
        "description": "幅広い分野の一般的な質問に答える",
        "details": "幅広い分野の一般的な質問に対して、正確で分かりやすい回答を提供してください。"
    },
    "2": {
        "name": "生成AI製品エキスパート",
        "description": "生成AIや関連製品、技術に関する専門的な質問に答える",
        "details": "生成AIや関連製品、技術に関する専門的な質問に対して、最新の情報と深い洞察を提供してください。"
    },
    "3": {
        "name": "カウンセラー",
        "description": "個人的な悩みや心理的な問題に対してサポートを提供する",
        "details": "個人的な悩みや心理的な問題に対して、共感的で支援的な回答を提供し、可能であれば適切なアドバイスも行ってください。"
    }
}

# ---------------------------
# State 定義
# ---------------------------
class State(BaseModel):
    query: str = Field(..., description="ユーザーからの質問")
    current_role: str = Field(default="", description="選定された回答ロール（ラベル名）")
    messages: Annotated[list[str], operator.add] = Field(default_factory=list, description="回答履歴")
    current_judge: bool = Field(default=False, description="品質チェックの結果 True/False")
    judgement_reason: str = Field(default="", description="品質チェックの判定理由")
    retry_count: int = Field(default=0, description="やり直し回数")
    max_retries: int = Field(default=2, description="やり直しの上限回数")

# ---------------------------
# LLM 初期化
# ---------------------------
# ※モデル名は手元の契約/環境に合わせて変更OK（例: "gpt-4o-mini"）
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = llm.configurable_fields(max_tokens=ConfigurableField(id="max_tokens"))

# ---------------------------
# ノード: ロール選択
# ---------------------------
selection_prompt = ChatPromptTemplate.from_template(
    """
あなたは質問応答システムのロール選択器です。
次のユーザー質問に最も適したロール番号を **半角数字1つ（1/2/3）だけ** 出力してください。説明や語尾は不要です。

ユーザーの質問:
{query}

ロールの選択肢:
1: {r1_name}: {r1_desc}
2: {r2_name}: {r2_desc}
3: {r3_name}: {r3_desc}
""".strip()
)
selection_chain = selection_prompt | llm.with_config(configurable={"max_tokens": 3}) | StrOutputParser()

def _pick_role_number(raw: str) -> str:
    m = re.search(r"[123]", raw)
    return m.group(0) if m else "1"

def selection_node(state: State) -> dict[str, Any]:
    raw = selection_chain.invoke({
        "query": state.query,
        "r1_name": ROLES["1"]["name"], "r1_desc": ROLES["1"]["description"],
        "r2_name": ROLES["2"]["name"], "r2_desc": ROLES["2"]["description"],
        "r3_name": ROLES["3"]["name"], "r3_desc": ROLES["3"]["description"],
    })
    role_num = _pick_role_number(raw.strip())
    selected_role = ROLES.get(role_num, ROLES["1"])["name"]
    return {"current_role": selected_role}

# ---------------------------
# ノード: 回答（初回 & 再回答）
# ---------------------------
answer_prompt = ChatPromptTemplate.from_template(
    """
あなたは「{role}」として回答します。

役割の詳細:
{role_details}

ユーザーの質問:
{query}

{feedback_block}

制約:
- 箇条書きや段落を使い、簡潔かつ根拠のある説明を心がける
- 不明点は推測せず、その旨を明示して代替案や次のアクションを提示する

最終回答を出力してください。
""".strip()
)
answer_chain = answer_prompt | llm | StrOutputParser()

def answering_node(state: State) -> dict[str, Any]:
    # 直前のチェックでNGだった場合は、フィードバックを渡して改善指示
    feedback_block = ""
    if state.retry_count > 0 and not state.current_judge and state.judgement_reason:
        feedback_block = f"前回の指摘点（品質改善のため必ず反映）:\n- {state.judgement_reason}\n"

    role_details = "\n".join([f"- {v['name']}: {v['details']}" for v in ROLES.values()])
    answer = answer_chain.invoke({
        "role": state.current_role or ROLES["1"]["name"],
        "role_details": role_details,
        "query": state.query,
        "feedback_block": feedback_block
    })
    # 履歴に追加
    return {"messages": [answer]}

# ---------------------------
# ノード: 品質チェック
# ---------------------------
class Judgement(BaseModel):
    reason: str = Field(default="", description="判定理由")
    judge: bool = Field(default=False, description="品質チェックの結果")

check_prompt = ChatPromptTemplate.from_template(
    """
以下の回答の品質を評価してください。

評価方針:
- 質問に直接的・正確に答えているか
- 誤情報・曖昧な断定がないか
- 構成が明瞭で読みやすいか
- 役割（{role}）として妥当か

出力は JSON で返してください:
- judge: 問題なければ true、改善が必要なら false
- reason: 判断理由を簡潔に

ユーザーの質問:
{query}

対象の回答:
{answer}
""".strip()
)
check_chain = check_prompt | llm.with_structured_output(Judgement)

def check_node(state: State) -> dict[str, Any]:
    answer = state.messages[-1] if state.messages else ""
    result: Judgement = check_chain.invoke({
        "query": state.query,
        "answer": answer,
        "role": state.current_role or ROLES["1"]["name"]
    })
    return {"current_judge": result.judge, "judgement_reason": result.reason}

# ---------------------------
# グラフ構築
# ---------------------------
workflow = StateGraph(State)
workflow.add_node("selection", selection_node)
workflow.add_node("answering", answering_node)
workflow.add_node("check", check_node)

workflow.set_entry_point("selection")
workflow.add_edge("selection", "answering")
workflow.add_edge("answering", "check")

# 条件分岐: 合格→END、不合格→（リトライ残あり）answering、（リトライ上限）END
def route_from_check(state: State) -> str:
    if state.current_judge:
        return END
    # 不合格の場合
    if state.retry_count < state.max_retries:
        # 次の answering に備えてカウンタを +1
        state.retry_count += 1  # LangGraphではstateの返却で書き戻しされる想定
        return "answering"
    # 上限に達したら終了
    return END

workflow.add_conditional_edges("check", route_from_check, {END: END, "answering": "answering"})
compiled_workflow = workflow.compile()

# ---------------------------
# 実行
# ---------------------------
if __name__ == "__main__":
    user_query = input("質問を入力してください: ")
    initial_state = State(query=user_query)
    result = compiled_workflow.invoke(initial_state)

    print(f"\n選択されたロール: {result['current_role']}")
    print(f"品質チェック: {result['current_judge']}（理由: {result['judgement_reason']}）")
    print("\n回答:")
    print(result["messages"][-1])
