import operator
from typing import Annotated, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.runnables import ConfigurableField
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


# ロールの定義
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


# Stateの定義
class State(BaseModel):
    query: str = Field(
       ..., description="ユーザーからの質問"
    )
    current_role: str = Field(
        default="", description="選定された回答ロール"
    )
    messages: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="回答履歴"
    )
    current_judge: bool = Field(
        default=False, description="品質チェックの結果"
    )
    judgement_reason: str = Field(
        default="", description="品質チェックの判定理由"
    )


# Chat modelの初期化
llm = ChatOpenAI(model="gpt-4", temperature=0)
llm = llm.configurable_fields(max_tokens=ConfigurableField(id="max_tokens"))


# ノードの定義
# selectionノードの実装
def selection_node(state: State) -> dict[str, Any]:
    query = state.query
    role_options = "\n".join([f"{k}: {v['name']}: {v['description']}" for k, v in ROLES.items()])
    prompt = ChatPromptTemplate.from_template(
        f"""
        あなたは質問応答システムの一部です。以下のユーザーからの質問に最も適したロールを選んでください。

        ユーザーの質問: {query}

        ロールの選択肢:
        {role_options}

        最も適切なロールの番号を一つだけ選んでください。
        """.strip()
    )
    chain = prompt | llm.with_config(configurable=dict(max_tokens=1)) | StrOutputParser()
    role_number = chain.invoke({"role_options": role_options, "query": query})

    selected_role = ROLES.get(role_number.strip(), ROLES["1"])["name"]
    return {"current_role": selected_role}
    

# answeringノードの実装
def answering_node(state: State) -> dict[str, Any]:
    query = state.query
    role = state.current_role
    role_details = "\n".join([f"- {v['name']}: {v['details']}" for v in ROLES.values()])
    prompt = ChatPromptTemplate.from_template(
        f"""
        あなたは{role}として回答してください。以下の質問に対して、あなたの役割に基づいた適切な回答を提供してください。
        
        役割の詳細：
        {role_details}
        
        質問：{query}

        ユーザーの質問: {query}

        ロールの選択肢:
        {role_details}
        
        回答：""".strip()
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"role": role,"role_details": role_details}, {"query": query})
    return {"messages": [answer]}



# checkノードの実装
class Judgement(BaseModel):
    reason: str = Field(default="", description="判定理由")
    Judge: bool = Field(default=False, description="品質チェックの結果")
    
def check_node(state: State) -> dict[str, Any]:
    query = state.query
    answer = state.messages[-1]
    prompt = ChatPromptTemplate.from_template(
        f"""
        以下の回答の品質をチェックし。問題がある場合は'False'、問題がない場合は'True'を返してください。
        また、その判定理由も説明してください。

        ユーザーの質問: {query}

        回答: {answer}
        """.strip()
    )
    chain = prompt | llm.with_structured_output(Judgement)
    result: Judgement = chain.invoke({"query": query}, {"answer": answer})
    return {"current_judge": result.Judge, "judgement_reason": result.reason}

# グラフの作成
workflow = StateGraph(State)

# ノードの追加
workflow.add_node("selection", selection_node)
workflow.add_node("answering", answering_node)
workflow.add_node("check", check_node)

# エッジの追加
workflow.set_entry_point("selection")
workflow.add_edge("selection", "answering")
workflow.add_edge("answering", "check")

# 条件付きエッジの定義
workflow.add_conditional_edges(
    "check",
    lambda state: state.current_judge,
    {True: END, False: "answering"}
)

# グラフのコンパイル
compiled_workflow = workflow.compile()

if __name__ == "__main__":
    user_query = input("質問を入力してください: ")
    initial_state = State(query=user_query)
    result = compiled_workflow.invoke(initial_state)

    print(f"\n選択されたロール: {result['current_role']}")
    print("\n回答:")
    print(result["messages"][-1])
