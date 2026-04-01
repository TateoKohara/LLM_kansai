# 大阪弁特化ローカルLLM — プロジェクト計画

## 概要

Qwen3-Swallow-8B-SFTをベースに、既存日本語会話データを大阪弁に変換した合成データ＋ITA_KANSAI_CORPUSでQLoRA SFTを実施。M5 Pro（48GB）上で`mlx-lm`を使い学習・推論し、HuggingFaceに公開する。モデルは**常に大阪弁で回答**する。

**リポジトリ**: https://github.com/TateoKohara/LLM_kansai

## 決定事項

| 項目 | 決定 |
|---|---|
| HW | Apple Silicon M5 Pro / 48GB |
| 方言範囲 | **大阪弁に限定**（京都弁・神戸弁は除外） |
| 応答戦略 | **常に大阪弁で回答** |
| ベースモデル | `tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2` (Apache 2.0) |
| 学習手法 | QLoRA via mlx-lm |
| データ戦略 | 複合C: 合成データ主体 + ITA_KANSAI_CORPUSを品質基準に |
| 合成データ変換 | ベースモデル（Qwen3-Swallow-8B 4bit）でローカル変換 |
| データ量 | v4: 1,714件（train） / 214件（valid） / 215件（test） |
| 公開先 | HuggingFace |

---

## Phase 1: データ準備 ✅ 完了

### Step 1.1: ソースデータセット収集 ✅
- 日本語会話データセット（instruction-response形式）を収集
  - `kunishou/databricks-dolly-15k-ja` — Dolly日本語版
  - `shi3z/alpaca_cleaned_ja` — 日本語Alpaca
- `scripts/step1_1_collect_data.py` で実行

### Step 1.2: ITA_KANSAI_CORPUS分析・大阪弁パターン辞書構築 ✅
- `joumonsugi/ITA_KANSAI_CORPUS`（MIT License）からパターン抽出
  - 語尾: 〜やで、〜やねん、〜やろ、〜してんか、〜やんか 等
  - 語彙: めっちゃ、ほんま、あかん、なんでやねん、ちゃう、おもろい 等
  - 感情差分リスト（ANGER/LAUGH/SAD/SURPRISE）から感情表現パターン
- 京都弁（〜どす、〜え）等を**明示的に除外**するガイドライン作成済み
- `scripts/step1_2_build_osaka_patterns.py` → `data/osaka_patterns.json`

### Step 1.3: 大阪弁変換パイプライン構築 ✅
- ベースモデル（Qwen3-Swallow-8B 4bit）でローカルバッチ変換
  - システムプロンプトに大阪弁パターン辞書を挿入
  - **assistantの応答のみ**を大阪弁化（userメッセージは標準語のまま）
  - 京都弁・神戸弁が混入しないよう明示指示
- `scripts/step1_3_osaka_convert.py` / `scripts/step1_3_local_convert.py`

### Step 1.4: データ品質管理・JSONL整形 ✅
- mlx-lm `chat`形式JSONL:
  ```json
  {"messages": [{"role":"system","content":"あんたは大阪弁で話す気さくなアシスタントやで。どんな質問にも大阪弁で答えてな。"},{"role":"user","content":"..."},{"role":"assistant","content":"（大阪弁応答）"}]}
  ```
- `train.jsonl` / `valid.jsonl` / `test.jsonl` に **8:1:1** 分割
- `scripts/step1_4_format_data.py`

### データ品質イテレーション（v1→v4）

| バージョン | 件数(train) | 主な改善 |
|---|---|---|
| v1 (osaka_data) | 2,818 | 初版 |
| v2 (osaka_data_clean) | 2,518 | 低品質除去（完全標準語、変換漏れ矢印等） |
| v3 | 1,714 | やで語尾のみ長文・残存標準語を除去（純度89.6%→99.2%） |
| **v4** (最終) | **1,714** | **やで偏重解消+語彙多様化（やで偏重58.3%→11.6%, 多様19.3%）** |

- v3→v4変換: `scripts/enhance_diversity.py`
  - やで語尾→多様語尾（やねん25%, やな20%, やで20%, ねん10%…の重み付き置換）
  - 標準語→大阪弁語彙（25+ルール）
  - 接続詞→大阪弁接続詞
- v2→v3変換: `scripts/clean_data_v2.py`

---

## Phase 2: ベースモデル・環境構築 ✅ 完了

### Step 2.1: ベースモデル
- **`tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2`**
  - Qwen3ベース 8B（36層, hidden_size 4096） / 日本語SFT済み / Apache 2.0
  - 4bit量子化: `uv run mlx_lm.convert --model tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2 -q`
  - `./mlx_model/` に格納（4.3GB）

### Step 2.2: 環境構築
- Python 3.12 + uv (`uv init --python 3.12` + `uv add "mlx-lm[train]"`)
- mlx-lm v0.31.1 / mlx v0.31.1

---

## Phase 3: QLoRA学習 ✅ 完了

### Step 3.1: 学習設定（実績値）

| パラメータ | 値 | 備考 |
|---|---|---|
| batch-size | **2** | メモリ効率 |
| num-layers | **16** | 全デフォルト層数 |
| grad-accumulation-steps | **4** | 実効batch-size 8 |
| grad-checkpoint | **有効** | メモリ節約 |
| max-seq-length | **2048** | |
| iters | **2000** | |
| learning-rate | **1e-5** | SFT済みモデルへの追加学習 |
| mask-prompt | **有効** | 応答部分のみに損失計算 |
| LoRA rank | **8** | scale=20.0, dropout=0.0 |
| 学習可能パラメータ | **9.699M** | 全体の0.118% |
| ピークメモリ | **11.3GB** | |

### Step 3.2: 学習実行（v4最終版）
```bash
uv run mlx_lm.lora \
  --model ./mlx_model \
  --train \
  --data ./data/osaka_data_v4 \
  --batch-size 2 \
  --num-layers 16 \
  --iters 2000 \
  --mask-prompt \
  --grad-accumulation-steps 4 \
  --grad-checkpoint \
  --max-seq-length 2048 \
  --learning-rate 1e-5 \
  --save-every 100 \
  --steps-per-eval 200 \
  --adapter-path ./adapters
```

#### 学習履歴

| 版 | データ | Train Loss | Val Loss | 備考 |
|---|---|---|---|---|
| v1 | osaka_data (2,818件) | 1.208 | 1.100 | 初版 |
| v2 | osaka_data_clean (2,518件) | 1.216 | 1.141 | クリーンデータ |
| **v4** | **osaka_data_v4 (1,714件)** | **1.195** | **1.547** | **多様性強化・最終版** |

- v4 Best Val loss: **iter1600で1.386**
- アダプタ: `adapters/`（v4 iter2000）, `adapters_v4_1600/`（v4 iter1600）

### Step 3.3: 評価結果 ✅

#### パープレキシティ（v4 test.jsonl）
| モデル | PPL |
|---|---|
| ベース | 12.1 |
| v2 | 7.4 |
| v4 (iter1600) | 7.7 |
| v4 (iter2000) | 8.0 |

#### 定性評価（20問自動採点）
| 指標 | ベース | v2 | v4(iter1600) | **v4(iter2000)** |
|---|---|---|---|---|
| 大阪弁ヒット | 17 | 31 | 35 | **41** |
| 汚染ヒット | 23 | 1 | 0 | **0** |
| 大阪弁純度 | 42.5% | 96.9% | 100% | **100%** |
| 平均応答長 | 677字 | 63字 | 59字 | **94字** |
| 語尾多様性 | — | やで74% | 10種 | **10種** |

**v4 iter2000が総合ベスト**: 純度100%, 汚染0, 多様性10種, 応答長94字

- 評価スクリプト: `scripts/eval_v4.py`
- 生成設定: temp=0.6, top_p=0.95, top_k=20, repetition_penalty=1.2

---

## Phase 4: モデル公開 ⬜ 未着手

### Step 4.1: アダプタ融合・アップロード
```bash
uv run mlx_lm.fuse \
  --model ./mlx_model \
  --adapter-path ./adapters \
  --upload-repo TateoKohara/Osaka-Swallow-8B \
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

## プロジェクト構成

```
LLM_kansai/
├── mlx_model/              # ベースモデル (4.3GB, .gitignore)
├── adapters/               # v4 iter2000 アダプタ (777MB, .gitignore)
├── adapters_v4_1600/       # v4 iter1600 アダプタ (37MB, .gitignore)
├── data/
│   ├── osaka_data_v4/      # 最終学習データ (train/valid/test.jsonl)
│   ├── ita_kansai_corpus/  # 参照コーパス (MIT License)
│   ├── osaka_patterns.json # 大阪弁パターン辞書
│   └── osaka_style_guide.md
├── scripts/
│   ├── step1_1_collect_data.py
│   ├── step1_2_build_osaka_patterns.py
│   ├── step1_3_osaka_convert.py
│   ├── step1_3_local_convert.py
│   ├── step1_4_format_data.py
│   ├── clean_data_v2.py     # v2→v3変換
│   ├── enhance_diversity.py  # v3→v4多様性強化
│   └── eval_v4.py            # 包括的評価
├── main.py
├── pyproject.toml
├── AGENTS.md
└── README.md
```

## 関連リソース
- `ml-explore/mlx-lm` — 学習・推論・融合・アップロードの主要ツール
- `tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2` — ベースモデル
- `joumonsugi/ITA_KANSAI_CORPUS` — 大阪弁参照コーパス (MIT License)
