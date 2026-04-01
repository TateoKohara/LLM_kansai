# 大阪弁特化ローカルLLM — プロジェクト計画

## 概要

Qwen3-Swallow-8B-SFTをベースに、既存日本語会話データを大阪弁に変換した合成データ＋ITA_KANSAI_CORPUSでQLoRA SFTを実施。M5 Pro（48GB）上で`mlx-lm`を使い学習・推論し、HuggingFaceに公開する。モデルは**常に大阪弁で回答**する。

## 決定事項

| 項目 | 決定 |
|---|---|
| HW | Apple Silicon M5 Pro / 48GB |
| 方言範囲 | **大阪弁に限定**（京都弁・神戸弁は除外） |
| 応答戦略 | **常に大阪弁で回答** |
| ベースモデル | `tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2` (Apache 2.0) |
| 学習手法 | QLoRA via mlx-lm |
| データ戦略 | 複合C: 合成データ主体 + ITA_KANSAI_CORPUSを品質基準に |
| 合成データAPI | 少量テスト（GPT-4o / Claude 各50件）で比較後に決定 |
| データ量 | 5,000件スタート → 不足なら追加（イテレーティブ） |
| 公開先 | HuggingFace |

## AIエージェントパイプライン（推奨3段階）

| 段階 | モデル | 役割 |
|---|---|---|
| Plan | Claude Opus 4.6 | 設計・計画 |
| Coding + Review | GPT-5.4 (Xhigh) 実装 → Claude Opus 4.6 レビュー | 異なるモデルでクロスチェック |
| Final QA | Claude Opus 4.6 | 最終確認・修正指示 |

---

## Phase 1: データ準備（最重要・最大工数）

### Step 1.1: ソースデータセット収集
- 日本語会話データセット（instruction-response形式）を収集
  - `tokyotech-llm/lmsys-chat-1m-synth` — 約30万件の日本語会話（Swallow SFTデータ）
  - `kunishou/databricks-dolly-15k-ja` — Dolly日本語版（約15k件）
  - `shi3z/alpaca_cleaned_ja` — 日本語Alpaca
- 目標: **5,000件の会話ペア**（多様なトピック・応答長をカバーするよう層化抽出）

### Step 1.2: ITA_KANSAI_CORPUS分析・大阪弁パターン辞書構築（parallel with 1.1）
- `joumonsugi/ITA_KANSAI_CORPUS`（MIT License）からパターン抽出
  - 語尾: 〜やで、〜やねん、〜やろ、〜してんか、〜やんか 等
  - 語彙: めっちゃ、ほんま、あかん、なんでやねん、ちゃう、おもろい 等
  - 感情差分リスト（ANGER/LAUGH/SAD/SURPRISE）から感情表現パターン
- 京都弁（〜どす、〜え）等を**明示的に除外**するガイドラインも作成

### Step 1.3: 大阪弁変換パイプライン構築（depends on 1.1, 1.2）
- 冒頭で GPT-4o / Claude 各50件ずつ変換テスト → 品質・コスト比較 → 本番API決定
- LLM APIでバッチ変換
  - システムプロンプトに大阪弁パターン辞書を挿入
  - **assistantの応答のみ**を大阪弁化（userメッセージは標準語のまま）
  - 京都弁・神戸弁が混入しないよう明示指示
- Pythonバッチ処理スクリプト（asyncio + API rate limiting）

### Step 1.4: データ品質管理・JSONL整形（depends on 1.3）
- サンプル100件を人手チェック（京都弁混入、不自然表現の確認）
- mlx-lm `chat`形式JSONL:
  ```json
  {"messages": [{"role":"system","content":"あんたは大阪弁で話す気さくなアシスタントやで。どんな質問にも大阪弁で答えてな。"},{"role":"user","content":"..."},{"role":"assistant","content":"（大阪弁応答）"}]}
  ```
- `train.jsonl` / `valid.jsonl` / `test.jsonl` に **8:1:1** 分割

---

## Phase 2: ベースモデル・環境構築（parallel with Phase 1後半）

### Step 2.1: ベースモデル
- **`tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2`**
  - Qwen3ベース 8B / 日本語SFT済み / Apache 2.0
  - mlx-lm LoRA対応済み（Qwen2ファミリー）

### Step 2.2: 環境構築 ✅ 完了
- Python 3.12 + uv (`uv init --python 3.12` + `uv add "mlx-lm[train]"`)
- mlx-lm v0.31.1 / mlx v0.31.1 インストール済み
- 4bit量子化: `uv run mlx_lm.convert --model tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2 -q`
- 推論テストでベースライン確認（大阪弁プロンプトへの現状の応答を記録）

---

## Phase 3: QLoRA学習（depends on Phase 1, 2）

### Step 3.1: 学習設定 — M5 Pro 48GBに最適化

| パラメータ | 値 | 根拠 |
|---|---|---|
| batch-size | **4** | 48GBなら余裕あり（デフォルト値） |
| num-layers | **16** | 48GBなら全デフォルト層数で問題なし |
| grad-accumulation-steps | **2** | 実効batch-size 8 |
| iters | **1500〜3000** | データ量次第で調整 |
| learning-rate | **1e-5** | 日本語SFT済みモデルへの追加学習のため低め |
| mask-prompt | **有効** | 応答部分のみに損失計算 |

### Step 3.2: 学習実行
```bash
mlx_lm.lora \
  --model ./mlx_model \
  --train \
  --data ./osaka_data \
  --batch-size 4 \
  --num-layers 16 \
  --iters 2000 \
  --mask-prompt \
  --grad-accumulation-steps 2
```

### Step 3.3: 評価
- test.jsonlでパープレキシティ比較（学習前 vs 後）
- 定性評価（同一プロンプト20件以上）:
  - 大阪弁で回答しているか
  - 京都弁・標準語が混ざっていないか
  - 応答の内容的な正確性は維持されているか
- 標準語タスクの性能劣化チェック（catastrophic forgetting確認）
- 不十分なら→イテレーション数増加 or データ追加して再学習

---

## Phase 4: モデル公開（depends on Phase 3）

### Step 4.1: アダプタ融合・アップロード
```bash
mlx_lm.fuse \
  --model ./mlx_model \
  --adapter-path ./adapters \
  --upload-repo <username>/Osaka-Swallow-8B \
  --hf-path tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2
```
- GGUF形式もエクスポート（`--export-gguf`）でllama.cpp互換

### Step 4.2: モデルカード
- ベースモデル帰属（Qwen3-Swallow / Apache 2.0）
- ITA_KANSAI_CORPUS MITライセンス表示（**表示義務あり**）
  ```
  Copyright (c) 2024 おふとんP, あみたろの声素材工房, Nacl_E
  This corpus is released under the MIT License.
  http://opensource.org/licenses/mit-license.php
  ```
- system prompt例・推奨生成パラメータ（Temperature=0.6, TopP=0.95, TopK=20）
- 使用例・制限事項

---

## 検証方法
1. ベースモデルのベースライン出力確認（大阪弁プロンプトで）
2. 学習後モデルで同じプロンプトの出力比較
3. test.jsonlパープレキシティ計測
4. 人手評価: 20〜50件のプロンプト出力を評価（可能なら大阪弁ネイティブ）
5. 標準語タスク（数学・コード）の性能劣化確認

## 関連リソース
- `ml-explore/mlx-lm` — 学習・推論・融合・アップロードの主要ツール
- `tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2` — ベースモデル
- `joumonsugi/ITA_KANSAI_CORPUS` — 大阪弁参照コーパス (MIT License)
