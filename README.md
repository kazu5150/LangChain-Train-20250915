# LangChain学習プロジェクト

LangChainの基本機能を学習するためのサンプルコード集です。

## 📁 プロジェクト構造

```
train-1/
├── LLM-Chat/           # LLMとの基本的なチャット機能
│   ├── chat.py         # メッセージ履歴を使った会話
│   ├── llm.py          # LLMの基本使用
│   ├── streaming.py    # ストリーミング応答
│   └── test.py         # テスト用ファイル
├── Template/           # プロンプトテンプレートの学習
│   ├── chat_prompt_template.py    # チャットプロンプトテンプレート
│   ├── malti_modal.py            # マルチモーダル機能
│   ├── message_placeholder.py    # メッセージプレースホルダー
│   └── prompt_template.py        # 基本プロンプトテンプレート
├── Output-Parser/      # 出力パーサーの学習
│   ├── str_output_parser.py      # 文字列出力パーサー
│   └── pydantic_output_parser.py # Pydantic出力パーサー
└── recipe.py           # 料理レシピ生成のサンプル
```

## 🚀 セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env.example`を参考に`.env`ファイルを作成し、OpenAI APIキーを設定してください。

```bash
cp .env.example .env
```

`.env`ファイルにAPIキーを追加：
```
OPENAI_API_KEY=your_api_key_here
```

## 📚 学習内容

### 1. LLMとの基本チャット (`LLM-Chat/`)
- **chat.py**: メッセージ履歴を管理した会話システム
- **llm.py**: LLMの基本的な使用方法
- **streaming.py**: リアルタイムストリーミング応答

### 2. プロンプトテンプレート (`Template/`)
- **chat_prompt_template.py**: 構造化されたチャットプロンプト
- **prompt_template.py**: 基本的なプロンプトテンプレート
- **malti_modal.py**: 画像やその他メディアを含むマルチモーダル処理

### 3. 出力パーサー (`Output-Parser/`)
- **str_output_parser.py**: 文字列形式での出力処理
- **pydantic_output_parser.py**: 構造化データ（JSON）としての出力処理

### 4. 実践例
- **recipe.py**: 料理レシピ生成の完全な例

## 🛠️ 使用方法

各ファイルは独立して実行できます：

```bash
# レシピ生成の例
python train-1/recipe.py

# 基本チャット
python train-1/LLM-Chat/chat.py

# 出力パーサーのテスト
python train-1/Output-Parser/str_output_parser.py
```

## 📦 主な依存関係

- `langchain`: メインフレームワーク
- `langchain-openai`: OpenAI統合
- `openai`: OpenAI APIクライアント
- `python-dotenv`: 環境変数管理
- `pydantic`: データバリデーション

## 🎯 学習目標

このプロジェクトでは以下のLangChainの概念を学習できます：

1. **LLM統合**: OpenAI GPTモデルとの連携
2. **プロンプトエンジニアリング**: 効果的なプロンプトの作成
3. **チェーン構築**: コンポーネントを組み合わせた処理フロー
4. **出力処理**: 構造化データとしての結果取得
5. **メッセージ管理**: 会話履歴の保持と活用

## 💡 次のステップ

- RAG（Retrieval-Augmented Generation）の実装
- カスタムチェーンの作成
- ベクトルデータベースとの統合
- エージェント機能の活用