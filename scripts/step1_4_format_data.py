#!/usr/bin/env python3
"""
Step 1.4: データ品質管理・JSONL整形
- 重複除去
- 品質フィルタ（大阪弁マーカー、標準語残留、短すぎ、英語混入）
- mlx-lm chat形式JSONL変換
- train.jsonl / valid.jsonl / test.jsonl 8:1:1 分割
"""

import json
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_FILE = DATA_DIR / "osaka_conversations.jsonl"
OUTPUT_DIR = DATA_DIR / "osaka_data"

SEED = 42
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

SYSTEM_PROMPT = "あんたは大阪弁で話す気さくなアシスタントやで。どんな質問にも大阪弁で答えてな。"

# 大阪弁マーカー（1つでも含まれていれば大阪弁と判定）
OSAKA_MARKERS = [
    "やで", "やねん", "やろ", "やな", "ねん", "へん",
    "めっちゃ", "ほんま", "あかん", "ちゃう", "やんか",
    "やわ", "せや", "ええ", "やから", "やけど", "おもろ",
    "しとる", "あんた", "ほな", "なんで",
]

# 京都弁マーカー（除外）
KYOTO_MARKERS = ["どす", "どすえ", "おこしやす", "おいでやす", "よろしおす"]

# 神戸弁マーカー（除外）
KOBE_MARKERS = ["しとう", "やっとう", "なんしよん"]


def has_osaka_marker(text: str) -> bool:
    return any(m in text for m in OSAKA_MARKERS)


def has_kyoto_marker(text: str) -> bool:
    return any(m in text for m in KYOTO_MARKERS)


def has_kobe_marker(text: str) -> bool:
    return any(m in text for m in KOBE_MARKERS)


def english_ratio(text: str) -> float:
    """英語文字の割合"""
    if not text:
        return 0.0
    eng = sum(1 for c in text if c.isascii() and c.isalpha())
    return eng / max(len(text), 1)


def quality_filter(row: dict) -> tuple[bool, str]:
    """品質フィルタ。(合格, 理由) を返す"""
    text = row.get("osaka_response", "")

    # 空・短すぎ
    if len(text.strip()) < 10:
        return False, "too_short"

    # 大阪弁マーカーなし
    if not has_osaka_marker(text):
        return False, "no_osaka_marker"

    # 京都弁混入
    if has_kyoto_marker(text):
        return False, "kyoto_mixed"

    # 神戸弁混入
    if has_kobe_marker(text):
        return False, "kobe_mixed"

    # 英語が多すぎる（30%以上）
    if english_ratio(text) > 0.3:
        return False, "too_much_english"

    # <think>タグ残留
    if "<think>" in text:
        return False, "think_tag"

    # 引用符で囲まれただけ（変換されていない可能性）
    stripped = text.strip().strip('"').strip("'")
    original = row.get("original_response", "")
    if stripped == original.strip():
        return False, "not_converted"

    return True, "pass"


def to_chat_format(row: dict) -> dict:
    """mlx-lm chat形式に変換"""
    # osaka_responseから余計な引用符を除去
    response = row["osaka_response"].strip()
    # 先頭・末尾の引用符除去
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1].strip()

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["instruction"].strip()},
            {"role": "assistant", "content": response},
        ]
    }


def main():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 読み込み
    with open(INPUT_FILE, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    print(f"読み込み: {len(rows)} 件")

    # 重複除去（最後のものを採用）
    deduped = {}
    for r in rows:
        deduped[r["index"]] = r
    rows = list(deduped.values())
    print(f"重複除去後: {len(rows)} 件")

    # 品質フィルタ
    passed = []
    reject_reasons = {}
    for r in rows:
        ok, reason = quality_filter(r)
        if ok:
            passed.append(r)
        else:
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    print(f"\n品質フィルタ通過: {len(passed)} 件 / {len(rows)} 件 ({len(passed)/len(rows)*100:.1f}%)")
    print("除外理由:")
    for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    # chat形式変換
    chat_data = [to_chat_format(r) for r in passed]

    # シャッフル
    random.shuffle(chat_data)

    # 分割
    n = len(chat_data)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)
    # 残りをtest
    train_data = chat_data[:n_train]
    valid_data = chat_data[n_train : n_train + n_valid]
    test_data = chat_data[n_train + n_valid :]

    print(f"\n分割:")
    print(f"  train: {len(train_data)}")
    print(f"  valid: {len(valid_data)}")
    print(f"  test:  {len(test_data)}")

    # 書き出し
    for name, data in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
        path = OUTPUT_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  💾 {path} ({len(data)} 件)")

    # サンプル表示
    print("\n=== train.jsonl サンプル ===")
    for item in train_data[:3]:
        msgs = item["messages"]
        print(f"\n  system: {msgs[0]['content'][:50]}")
        print(f"  user:   {msgs[1]['content'][:80]}")
        print(f"  asst:   {msgs[2]['content'][:80]}")

    print(f"\n✅ Step 1.4 完了: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
