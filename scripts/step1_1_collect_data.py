#!/usr/bin/env python3
"""
Step 1.1: ソースデータセット収集
- 日本語会話データセット（instruction-response形式）をHuggingFaceからダウンロード
- 共通形式に正規化
- 応答長に基づく層化抽出で5,000件を選択
- data/raw_conversations.jsonl に保存
"""

import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "raw_conversations.jsonl"
TARGET_COUNT = 5000
SEED = 42


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def is_japanese(text: str, threshold: float = 0.05) -> bool:
    """テキストが日本語を含むか簡易判定"""
    if not text:
        return False
    jp = sum(
        1 for c in text
        if ("\u3040" <= c <= "\u309F")    # ひらがな
        or ("\u30A0" <= c <= "\u30FF")    # カタカナ
        or ("\u4E00" <= c <= "\u9FFF")    # CJK漢字
    )
    return jp / max(len(text), 1) > threshold


def quality_filter(instruction: str, response: str) -> bool:
    """最低限の品質フィルタ"""
    if len(instruction.strip()) < 5 or len(response.strip()) < 10:
        return False
    # 極端に長い応答は除外（トークン数爆発対策）
    if len(response) > 5000:
        return False
    return True


# ---------------------------------------------------------------------------
# データローダー
# ---------------------------------------------------------------------------

def load_dolly_ja() -> list[dict]:
    """kunishou/databricks-dolly-15k-ja"""
    print("📦 Loading kunishou/databricks-dolly-15k-ja ...")
    try:
        ds = load_dataset("kunishou/databricks-dolly-15k-ja", split="train")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return []

    print(f"  columns: {ds.column_names}, rows: {len(ds)}")
    pairs = []
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        output = (row.get("output") or "").strip()
        if not instruction or not output:
            continue
        if inp:
            instruction = f"{instruction}\n{inp}"
        if not quality_filter(instruction, output):
            continue
        pairs.append({
            "instruction": instruction,
            "response": output,
            "source": "dolly-ja",
        })
    print(f"  ✅ {len(pairs)} pairs extracted")
    return pairs


def load_alpaca_ja() -> list[dict]:
    """shi3z/alpaca_cleaned_ja_json"""
    print("📦 Loading shi3z/alpaca_cleaned_ja_json ...")
    try:
        ds = load_dataset("shi3z/alpaca_cleaned_ja_json", split="train")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return []

    print(f"  columns: {ds.column_names}, rows: {len(ds)}")
    pairs = []
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        output = (row.get("output") or "").strip()
        if not instruction or not output:
            continue
        if inp:
            instruction = f"{instruction}\n{inp}"
        if not quality_filter(instruction, output):
            continue
        pairs.append({
            "instruction": instruction,
            "response": output,
            "source": "alpaca-ja",
        })
    print(f"  ✅ {len(pairs)} pairs extracted")
    return pairs


def load_lmsys_synth() -> list[dict]:
    """tokyotech-llm/lmsys-chat-1m-synth — GPT-OSS subset (Apache 2.0) を直接ダウンロード"""
    print("📦 Loading tokyotech-llm/lmsys-chat-1m-synth (GPT-OSS subset) ...")
    filename = "gpt-oss-lmsys-chat-1m-synth-ja+en.jsonl.gz"
    try:
        path = hf_hub_download(
            repo_id="tokyotech-llm/lmsys-chat-1m-synth",
            filename=filename,
            repo_type="dataset",
        )
    except Exception as e:
        print(f"  ⚠️ Skipping (may require auth): {e}")
        return []

    pairs = []
    scanned = 0
    print(f"  Parsing {filename} ...")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            scanned += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            # synthesized_multiturn_conversation または他の会話列を試す
            messages = (
                row.get("synthesized_multiturn_conversation")
                or row.get("messages")
                or row.get("conversation")
            )
            if messages and isinstance(messages, list):
                user_msg, asst_msg = None, None
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role") or msg.get("from") or ""
                    content = msg.get("content") or msg.get("value") or ""
                    if role in ("user", "human") and not user_msg:
                        user_msg = content
                    elif role in ("assistant", "gpt") and not asst_msg:
                        asst_msg = content
                if user_msg and asst_msg and is_japanese(asst_msg):
                    if quality_filter(user_msg, asst_msg):
                        pairs.append({
                            "instruction": user_msg.strip(),
                            "response": asst_msg.strip(),
                            "source": "lmsys-synth",
                        })
            else:
                # synthesized_assistant_responses (リスト) から最初の日本語応答を使う
                responses = row.get("synthesized_assistant_responses", [])
                if isinstance(responses, list) and responses:
                    for resp in responses:
                        if isinstance(resp, str) and is_japanese(resp):
                            # conversation_id から元の指示を推定できないため skip
                            break

            if len(pairs) >= 5000:
                break
            if scanned % 50_000 == 0:
                print(f"    scanned {scanned:,} rows → {len(pairs)} Japanese pairs")

    print(f"  ✅ {len(pairs)} pairs extracted (scanned {scanned:,} rows)")
    return pairs


def load_alpaca_gpt4_ja() -> list[dict]:
    """FreedomIntelligence/alpaca-gpt4-japanese — 代替・多様性用"""
    print("📦 Loading FreedomIntelligence/alpaca-gpt4-japanese ...")
    try:
        ds = load_dataset("FreedomIntelligence/alpaca-gpt4-japanese", split="train")
    except Exception as e:
        print(f"  ⚠️ Skipping: {e}")
        return []

    print(f"  columns: {ds.column_names}, rows: {len(ds)}")
    pairs = []
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        output = (row.get("output") or "").strip()
        if not instruction or not output:
            continue
        if inp:
            instruction = f"{instruction}\n{inp}"
        if not quality_filter(instruction, output):
            continue
        pairs.append({
            "instruction": instruction,
            "response": output,
            "source": "alpaca-gpt4-ja",
        })
    print(f"  ✅ {len(pairs)} pairs extracted")
    return pairs


# ---------------------------------------------------------------------------
# 層化抽出
# ---------------------------------------------------------------------------

def stratified_sample(all_pairs: list[dict], target: int) -> list[dict]:
    """応答長に基づく層化抽出"""
    buckets: dict[str, list[int]] = {"short": [], "medium": [], "long": []}
    for idx, pair in enumerate(all_pairs):
        n = len(pair["response"])
        if n < 100:
            buckets["short"].append(idx)
        elif n < 500:
            buckets["medium"].append(idx)
        else:
            buckets["long"].append(idx)

    total = sum(len(b) for b in buckets.values())
    if total == 0:
        return []

    sampled: set[int] = set()
    for name, indices in buckets.items():
        if not indices:
            continue
        n = max(200, int(target * len(indices) / total))
        n = min(n, len(indices))
        chosen = random.sample(indices, n)
        sampled.update(chosen)
        print(f"  {name}: {len(indices):,} available → sampled {len(chosen):,}")

    # 不足分を補充
    if len(sampled) < target:
        remaining = [i for i in range(len(all_pairs)) if i not in sampled]
        extra = min(target - len(sampled), len(remaining))
        sampled.update(random.sample(remaining, extra))

    # 超過分をランダム削減
    if len(sampled) > target:
        sampled = set(random.sample(list(sampled), target))

    result = [all_pairs[i] for i in sorted(sampled)]
    random.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs: list[dict] = []

    all_pairs.extend(load_dolly_ja())
    all_pairs.extend(load_alpaca_ja())
    all_pairs.extend(load_alpaca_gpt4_ja())
    all_pairs.extend(load_lmsys_synth())

    print(f"\n{'='*50}")
    print(f"Total pairs collected: {len(all_pairs):,}")
    for src, cnt in Counter(p["source"] for p in all_pairs).most_common():
        print(f"  {src}: {cnt:,}")

    if len(all_pairs) < TARGET_COUNT:
        print(f"⚠️ Only {len(all_pairs)} pairs available (target: {TARGET_COUNT})")

    print(f"\nStratified sampling → {TARGET_COUNT} pairs")
    sampled = stratified_sample(all_pairs, TARGET_COUNT)
    print(f"Final sample: {len(sampled):,}")
    for src, cnt in Counter(p["source"] for p in sampled).most_common():
        print(f"  {src}: {cnt:,}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in sampled:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"\n💾 Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
