#!/usr/bin/env python3
"""
Step 1.3 (ローカル版): ベースモデル (Qwen3-Swallow-8B 4bit) で大阪弁バッチ変換
- mlx-lm の generate API を使用
- chat形式で system prompt + user prompt → assistant応答
- 5,000件をバッチ処理 (リジューム対応)
"""

import json
import re
import sys
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = Path(__file__).parent.parent / "mlx_model"
RAW_FILE = DATA_DIR / "raw_conversations.jsonl"
STYLE_GUIDE_FILE = DATA_DIR / "osaka_style_guide.md"
OUTPUT_FILE = DATA_DIR / "osaka_conversations.jsonl"


def load_style_guide() -> str:
    return STYLE_GUIDE_FILE.read_text(encoding="utf-8")


def build_system_prompt(style_guide: str) -> str:
    return f"""あなたは「標準語の日本語テキストを大阪弁に変換する」専門の翻訳者です。

## タスク
ユーザーから渡される「アシスタントの応答」を大阪弁に変換してください。

## ルール
1. 大阪弁のみを使用（京都弁・神戸弁は絶対に使わない）
2. 内容・情報の正確性はそのまま保つ
3. 自然な大阪弁の会話体にする（過度に誇張しない）
4. 固有名詞・数値・専門用語はそのまま残す
5. リスト形式の応答はリスト形式を維持する
6. 変換後の大阪弁テキストのみを出力する（説明や注釈は不要）

## 大阪弁スタイルガイド
{style_guide}"""


def build_user_prompt(instruction: str, response: str) -> str:
    return f"""以下のアシスタント応答を大阪弁に変換してください。変換後のテキストのみを出力してください。 /no_think

【ユーザーの質問（参考）】
{instruction}

【変換対象のアシスタント応答】
{response}"""


def strip_think_tags(text: str) -> str:
    """Qwen3の <think>...</think> タグを除去（閉じタグが無い場合も対応）"""
    # <think>...</think> ブロックを除去
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # </think>が無く<think>で始まる場合: <think>以降を全て除去して残りを使う
    if "<think>" in result:
        # <think>より前の部分があればそれを使う、なければ<think>以降から抽出
        before = result.split("<think>")[0].strip()
        if before:
            return before
        # <think>の後の内容から日本語テキスト部分を抽出
        after = result.split("<think>")[-1]
        # 末尾の</think>を除去
        after = after.replace("</think>", "")
        # 英語の思考部分をスキップして日本語テキストを探す
        lines = after.strip().splitlines()
        jp_lines = [l for l in lines if any(
            "\u3040" <= c <= "\u9FFF" for c in l
        ) and not l.strip().startswith(("Original", "We need", "The user", "Need to", "Let", "Must", "Convert"))]
        return "\n".join(jp_lines).strip()
    return result.strip()


def main():
    # --- モデルロード ---
    print(f"🔧 Loading model from {MODEL_PATH} ...")
    t0 = time.time()
    model, tokenizer = load(str(MODEL_PATH))
    print(f"   Loaded in {time.time()-t0:.1f}s")

    # --- データ読み込み ---
    with open(RAW_FILE, encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
    print(f"📂 Total data: {len(all_data)} entries")

    # --- リジューム対応 ---
    done_indices: set[int] = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                done_indices.add(row["index"])
        print(f"   Resume: {len(done_indices)} already done")

    remaining = [(i, d) for i, d in enumerate(all_data) if i not in done_indices]
    if not remaining:
        print("✅ 全件処理済みです")
        return

    print(f"🔄 Processing {len(remaining)} entries ...")

    # --- スタイルガイド & プロンプト ---
    style_guide = load_style_guide()
    system_prompt = build_system_prompt(style_guide)

    success = 0
    errors = 0
    start_time = time.time()

    # バッチ処理
    for batch_i, (idx, sample) in enumerate(remaining):
        user_prompt = build_user_prompt(sample["instruction"], sample["response"])

        # chat形式でプロンプト構築
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # フォールバック
            prompt_text = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        try:
            # 応答の長さは元の応答の1.5倍 + 余白
            max_tokens = min(max(int(len(sample["response"]) * 2), 512), 4096)

            raw_output = generate(
                model,
                tokenizer,
                prompt=prompt_text,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=0.7, top_p=0.95),
                verbose=False,
            )

            osaka_response = strip_think_tags(raw_output).strip()

            if not osaka_response:
                print(f"  ⚠️ [{idx}] Empty output, using original")
                osaka_response = sample["response"]
                errors += 1
            else:
                success += 1

            result = {
                "index": idx,
                "instruction": sample["instruction"],
                "original_response": sample["response"],
                "osaka_response": osaka_response,
                "source": sample["source"],
            }

            # 逐次書き込み
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"  ❌ [{idx}] Error: {e}")
            errors += 1

        # 進捗表示 (50件ごと)
        if (batch_i + 1) % 50 == 0 or batch_i == 0:
            elapsed = time.time() - start_time
            rate = (batch_i + 1) / max(elapsed, 1)
            eta = (len(remaining) - batch_i - 1) / max(rate, 0.001)
            print(
                f"  [{batch_i+1}/{len(remaining)}] "
                f"成功: {success}, エラー: {errors}, "
                f"{rate:.2f} 件/s, ETA: {eta/60:.1f}分"
            )

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"バッチ変換完了!")
    print(f"  成功: {success}, エラー: {errors}")
    print(f"  所要時間: {elapsed/60:.1f}分 ({elapsed/max(success,1):.1f}s/件)")
    print(f"  出力: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
