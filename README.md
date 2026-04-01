# 🏯 Osaka-Swallow-8B — 大阪弁特化LLM

関西弁、主に**大阪弁での会話**にフィーチャーした小型特化LLMです。  
標準語で質問しても、常に大阪弁で返答してくれる気さくなアシスタントを目指しています。

> **開発手法**: 本プロジェクトは **Claude Opus 4.6 によるvibe coding** で開発されました。  
> データパイプライン設計・スクリプト実装・学習パラメータ調整・評価まで、AIエージェントとの対話を通じて構築しています。

## 特徴

- 🗣️ **大阪弁に特化** — 京都弁・神戸弁を明示的に除外し、大阪弁の語尾・語彙を忠実に再現
- 🎯 **大阪弁純度100%** — 評価20問すべてで大阪弁のみの応答を達成（標準語汚染ゼロ）
- 📝 **語尾多様性10種** — やで / やねん / やな / ねん / やろ / やんか 等の自然な使い分け
- 🍎 **Apple Silicon最適化** — MLXフレームワークでM5 Pro 48GB上で学習・推論

## ベースモデル

| 項目 | 詳細 |
|---|---|
| モデル | [`tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2`](https://huggingface.co/tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2) |
| アーキテクチャ | Qwen3 8B (36層, hidden_size 4096) |
| ライセンス | Apache License 2.0 |
| 特徴 | 東京工業大学による日本語SFT済みモデル |

## 学習

### 手法
- **QLoRA** (Quantized Low-Rank Adaptation) via [mlx-lm](https://github.com/ml-explore/mlx-lm)
- LoRA rank=8, scale=20.0, 16層に適用
- 学習可能パラメータ: 9.699M（全体の0.118%）

### 学習データ
ベースモデル（Qwen3-Swallow-8B 4bit）自身を用いて、日本語会話データセットのassistant応答を大阪弁に変換した合成データ（1,714件）で学習。  
ITA_KANSAI_CORPUSから抽出した大阪弁パターン辞書をプロンプトに組み込むことで、変換品質を担保しています。

| パラメータ | 値 |
|---|---|
| データ件数 | train: 1,714 / valid: 214 / test: 215 |
| バッチサイズ | 2（grad-accum 4 → 実効8） |
| 学習率 | 1e-5 |
| イテレーション | 2,000 |
| 最大系列長 | 2,048 |
| ピークメモリ | 11.3GB |

### 評価結果

| 指標 | ベースモデル | **ファインチューン後** |
|---|---|---|
| 大阪弁純度 | 42.5% | **100%** |
| 標準語汚染 | 23件/20問 | **0件** |
| 語尾多様性 | — | **10種** |
| 平均応答長 | 677字 | **94字** |
| パープレキシティ | 12.1 | **8.0** |

## クイックスタート

### 推論

```bash
# 環境構築
uv sync

# 推論（ベースモデル + LoRAアダプタ）
uv run mlx_lm.generate \
  --model ./mlx_model \
  --adapter-path ./adapters \
  --prompt "あんたは大阪弁で話す気さくなアシスタントやで。どんな質問にも大阪弁で答えてな。\n\nユーザー: 自己紹介してください\nアシスタント:" \
  --temp 0.6 --top-p 0.95 --top-k 20 --repetition-penalty 1.2
```

### 学習の再現

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
  --adapter-path ./adapters
```

## プロジェクト構成

```
LLM_kansai/
├── mlx_model/              # ベースモデル 4bit量子化 (4.3GB, Git管理外)
├── adapters/               # v4 LoRAアダプタ (Git管理外)
├── data/
│   ├── osaka_data_v4/      # 学習データ (train/valid/test.jsonl)
│   ├── ita_kansai_corpus/  # 参照コーパス (MIT License)
│   ├── osaka_patterns.json # 大阪弁パターン辞書
│   └── osaka_style_guide.md
├── scripts/
│   ├── step1_1_collect_data.py       # データ収集
│   ├── step1_2_build_osaka_patterns.py # パターン辞書構築
│   ├── step1_3_osaka_convert.py      # API版大阪弁変換 (※未使用)
│   ├── step1_3_local_convert.py      # ベースモデルによるローカル変換 (実際に使用)
│   ├── step1_4_format_data.py        # JSONL整形
│   ├── clean_data_v2.py              # データクリーニング
│   ├── enhance_diversity.py          # 語尾多様性強化
│   └── eval_v4.py                    # 包括的評価
├── main.py
├── pyproject.toml
├── AGENTS.md               # プロジェクト計画・進捗
└── README.md
```

## 使用データセット・ライセンス

### ソースデータセット
| データセット | 用途 | ライセンス |
|---|---|---|
| [kunishou/databricks-dolly-15k-ja](https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja) | 日本語会話ベースデータ | CC BY-SA 3.0 |
| [shi3z/alpaca_cleaned_ja](https://huggingface.co/datasets/shi3z/alpaca_cleaned_ja) | 日本語会話ベースデータ | Apache License 2.0 |

### 参照コーパス

**ITA_KANSAI_CORPUS** — 大阪弁パターン辞書の構築に使用

```
MIT License

Copyright (c) 2024 おふとんP, あみたろの声素材工房, Nacl_E

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

- リポジトリ: [joumonsugi/ITA_KANSAI_CORPUS](https://github.com/joumonsugi/ITA_KANSAI_CORPUS)

### ベースモデル
- [tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2](https://huggingface.co/tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2) — Apache License 2.0

### 開発ツール
| ツール | 用途 |
|---|---|
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | QLoRA学習・推論・モデル融合 |
| Qwen3-Swallow-8B (ベースモデル 4bit) | 大阪弁変換（合成データ生成） |
| [Claude Opus 4.6](https://anthropic.com/) | Vibe coding（開発全般） |

> **注**: GPT-4o / Claude API による大阪弁変換スクリプト (`step1_3_osaka_convert.py`) もリポジトリに含まれていますが、実際の学習データ生成には使用していません。すべての変換はベースモデルによるローカル処理 (`step1_3_local_convert.py`) で実施しています。

## 動作要件

- Apple Silicon Mac（M1以降, 16GB+ RAM推奨）
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) パッケージマネージャ
