"""
大阪弁多様性強化スクリプト

v3データの「やで」偏重問題を解消し、語尾・語彙の多様性を向上させる。

戦略:
1. やで語尾の多様化: 文末の「やで」を文脈に応じて「やねん」「やな」「やろ」等に分散
2. 残存標準語の大阪弁化: LLM変換で漏れた標準語表現を機械的に置換
3. 大阪弁語彙注入: 「とても→めっちゃ」「本当→ほんま」等の追加
"""

import json
import random
import re
import copy
from pathlib import Path

random.seed(42)

# ========== 語尾多様化ルール ==========
# やで → 他の語尾にランダム置換（文脈ヒントで選択）
YADE_ALTERNATIVES = [
    ("やねん", 0.25),   # 説明・理由
    ("やな", 0.20),     # 感想・しみじみ
    ("やで", 0.20),     # そのまま（一部キープ）
    ("ねん", 0.10),     # 説明（軽め）
    ("やんか", 0.08),   # 確認・共感求め
    ("やろ", 0.07),     # 推量・反語
    ("やわ", 0.05),     # 柔らかい断定
    ("やんな", 0.05),   # 確認・同意求め
]

def weighted_choice(alternatives):
    """重み付きランダム選択"""
    total = sum(w for _, w in alternatives)
    r = random.random() * total
    cumul = 0
    for item, weight in alternatives:
        cumul += weight
        if r <= cumul:
            return item
    return alternatives[0][0]


def diversify_yade_endings(text):
    """文末の「やで」を多様な語尾に置換する"""
    # 文を分割（改行・句点区切り）
    lines = text.split("\n")
    new_lines = []
    
    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
        
        # 文末の「やで」「やで。」を検出して置換
        # ただし1文しかない行はそのまま（やで維持率高め）
        parts = re.split(r'(。)', line)
        reconstructed = []
        
        for i, part in enumerate(parts):
            if part == "。":
                reconstructed.append(part)
                continue
            
            stripped = part.rstrip()
            if stripped.endswith("やで"):
                new_ending = weighted_choice(YADE_ALTERNATIVES)
                stripped = stripped[:-2] + new_ending
                reconstructed.append(stripped + part[len(part.rstrip()):])
            else:
                reconstructed.append(part)
        
        new_lines.append("".join(reconstructed))
    
    return "\n".join(new_lines)


# ========== 語彙置換ルール ==========
# (regex_pattern, replacement, description)
VOCAB_REPLACEMENTS = [
    # 高頻度：安全な置換
    (r'とても(?=[^\w])', 'めっちゃ', '強調'),
    (r'すごく', 'めっちゃ', '強調'),
    (r'非常に', 'めっちゃ', '強調'),
    (r'本当に', 'ほんまに', '強調副詞'),
    (r'(?<![ほんまにっ])本当(?=[のにはがをもで])', 'ほんま', '名詞'),
    (r'ダメ(?=[やでだ。！\n])', 'あかん', '禁止'),
    (r'駄目(?=[やでだ。！\n])', 'あかん', '禁止'),
    (r'いけない', 'あかん', '禁止'),
    
    # 語尾系：残存する標準語を変換
    (r'です(?=[。\n！？])', 'や', '断定'),
    (r'ですね', 'やね', '同意'),
    (r'ですよ', 'やで', '強調'),
    (r'でしょう', 'やろ', '推量'),
    (r'ました(?=[。\n！？])', 'たわ', '過去'),
    (r'ます(?=[。\n！？])', 'るで', '丁寧→大阪弁'),
    (r'ますが', 'るけど', '逆接'),
    (r'ません(?=[。\n！？])', 'へん', '否定'),
    (r'ではない', 'ちゃう', '否定'),
    (r'じゃない', 'ちゃう', '否定'),
    (r'している(?=[。\n])', 'しとる', '進行'),
    (r'しています', 'してんねん', '進行+説明'),
    (r'できない', 'でけへん', '不可能'),
    (r'してください', 'してな', '依頼'),
    (r'しなければ', 'せなあかん', '義務'),
    (r'(?:なぜ|どうして)(?=[、。？])', 'なんで', '疑問'),
    
    # 中頻度：少し注意が必要
    (r'良い(?=[。やでねんな])', 'ええ', '形容詞'),
    (r'面白い', 'おもろい', '形容詞'),
    (r'つまらない', 'おもんない', '形容詞'),
    (r'(?:すみません|済みません)', 'すまんな', '謝罪'),
    (r'ありがとう(?:ございます)?', 'おおきに', '感謝'),
]


def apply_vocab_replacements(text):
    """残存標準語を大阪弁に置換"""
    for pattern, replacement, _ in VOCAB_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    return text


# ========== フィラー・表現注入 ==========
SENTENCE_STARTERS = [
    ("まず", "まずな、"),
    ("次に", "ほんで、"),
    ("それから", "ほんでな、"),
    ("しかし", "せやけど、"),
    ("でも", "せやけど、"),
    ("ただし", "ただな、"),
    ("つまり", "つまりな、"),
    ("要するに", "要はな、"),
    ("例えば", "例えばやけど、"),
]

def inject_connectives(text):
    """接続詞を大阪弁に変換"""
    for std, osaka in SENTENCE_STARTERS:
        # 文頭 or 改行直後の接続詞を置換
        text = re.sub(rf'(?:^|\n){std}[、,]?\s?', lambda m: m.group().replace(std + "、", osaka).replace(std + ",", osaka).replace(std, osaka), text)
    return text


def enhance_response(text):
    """assistant応答の大阪弁多様性を強化"""
    original = text
    
    # 1. やで語尾多様化
    text = diversify_yade_endings(text)
    
    # 2. 残存標準語→大阪弁
    text = apply_vocab_replacements(text)
    
    # 3. 接続詞の大阪弁化
    text = inject_connectives(text)
    
    return text


def process_file(input_path, output_path):
    """JSONL処理"""
    records = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line))
    
    enhanced = []
    for record in records:
        new_record = copy.deepcopy(record)
        for msg in new_record["messages"]:
            if msg["role"] == "assistant":
                msg["content"] = enhance_response(msg["content"])
        enhanced.append(new_record)
    
    with open(output_path, "w") as f:
        for record in enhanced:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Processed {len(enhanced)} records: {input_path} -> {output_path}")


def main():
    input_dir = Path("data/osaka_data_v3")
    output_dir = Path("data/osaka_data_v4")
    output_dir.mkdir(exist_ok=True)
    
    for split in ["train", "valid", "test"]:
        process_file(input_dir / f"{split}.jsonl", output_dir / f"{split}.jsonl")
    
    print("\n=== 完了 ===")
    print(f"出力先: {output_dir}")


if __name__ == "__main__":
    main()
