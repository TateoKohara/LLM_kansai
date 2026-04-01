"""データクリーニング v2: 深層分析で発見された問題を修正
問題カテゴリ:
  1. 完全標準語 (112件 = 4.4%) — 大阪弁パターンゼロ
  2. 語尾だけ「やで」(937件 = 37.2%) — 多様性不足の根本原因
  3. 変換漏れ矢印 (33件 = 1.3%) — => や -> が残存
  4. です/ます混入 (94件 = 3.7%) — 標準語が3回以上
  5. 英語命令文混入 — Thus, Actually, output only等

対策:
  A. 完全標準語・変換漏れ・高汚染を除去
  B. 語尾だけ「やで」のうち、短文（<50文字）はそのまま残す（「2やで。」等は正しい）
  C. 語尾だけ「やで」のうち、長文（>=50文字）は低品質として除去
  D. です/ます混入が3回以上は除去
"""
import json
import re
import random
import os

random.seed(42)

INPUT_DIR = "data/osaka_data_clean"
OUTPUT_DIR = "data/osaka_data_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 大阪弁パターン（「やで」以外のもの）
OSAKA_RICH = r"やねん|やろ|やな[。！\s、]|やん[。！？\s、]|やんか|めっちゃ|ほんま|あかん|ちゃう|おもろい|しとる|しとった|おんねん|せなあかん|でけへん|できひん|おる[。！\s、やで]|せやな|おおきに|ぎょうさん|しんどい|ええ[よやで。！]|してんか|してや[。！]|へん[。！\s、]"

# 汚染パターン
DESU_MASU = r"です[。！\s、]|ます[。！\s、]|です$|ます$"
ARROW = r"=>|-> "
ENGLISH_LEAK = re.compile(
    r"(?:Thus output|Actually |output only|Maybe\.|keep\.|paraphrase|=> Osaka|=> \")",
    re.IGNORECASE,
)

stats = {
    "total": 0,
    "kept": 0,
    "removed_full_standard": 0,
    "removed_yade_only_long": 0,
    "removed_arrow": 0,
    "removed_desu_masu": 0,
    "removed_english": 0,
}


def classify(asst_text):
    """応答を分類し、(keep: bool, reason: str)を返す"""
    # 大阪弁ヒット計算
    yade_hits = len(re.findall(r"やで[。！\s、]|やで$", asst_text))
    rich_hits = len(re.findall(OSAKA_RICH, asst_text))
    total_osaka = yade_hits + rich_hits
    dm_hits = len(re.findall(DESU_MASU, asst_text))

    # 変換漏れ矢印
    if re.search(ARROW, asst_text):
        return False, "arrow"

    # 英語命令文混入
    if ENGLISH_LEAK.search(asst_text):
        return False, "english"

    # 完全標準語（大阪弁ヒットゼロ）
    if total_osaka == 0:
        return False, "full_standard"

    # です/ます混入が3回以上
    if dm_hits >= 3:
        return False, "desu_masu"

    # 語尾だけ「やで」パターン（豊かな大阪弁表現がない）
    if rich_hits == 0 and yade_hits >= 1:
        # 短文は許容（「2やで。」「火山やで。」等）
        if len(asst_text) >= 100:
            return False, "yade_only_long"

    return True, "ok"


# 全データ読み込み（train + valid + test を統合して再分割）
all_data = []
for split in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
    path = os.path.join(INPUT_DIR, split)
    with open(path) as f:
        for line in f:
            stats["total"] += 1
            d = json.loads(line)
            asst = d["messages"][-1]["content"]
            keep, reason = classify(asst)
            if keep:
                all_data.append(d)
                stats["kept"] += 1
            else:
                stats[f"removed_{reason}"] += 1

print("=" * 60)
print("  データクリーニング v2 結果")
print("=" * 60)
print(f"  入力: {stats['total']}件")
print(f"  出力: {stats['kept']}件 ({stats['kept']/stats['total']*100:.1f}%)")
print(f"  除去内訳:")
print(f"    完全標準語:      {stats['removed_full_standard']:4d}件")
print(f"    やで語尾のみ長文: {stats['removed_yade_only_long']:4d}件")
print(f"    変換漏れ矢印:    {stats['removed_arrow']:4d}件")
print(f"    です/ます混入:   {stats['removed_desu_masu']:4d}件")
print(f"    英語命令文:      {stats['removed_english']:4d}件")

# シャッフル & 8:1:1分割
random.shuffle(all_data)
n = len(all_data)
n_train = int(n * 0.8)
n_valid = int(n * 0.1)
splits = {
    "train.jsonl": all_data[:n_train],
    "valid.jsonl": all_data[n_train:n_train + n_valid],
    "test.jsonl": all_data[n_train + n_valid:],
}

for fname, data in splits.items():
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  {fname}: {len(data)}件")

# 品質チェック
print(f"\n=== v3データ品質確認 ===")
osaka_total = 0
contam_total = 0
for d in all_data:
    asst = d["messages"][-1]["content"]
    osaka_total += len(re.findall(OSAKA_RICH + r"|やで[。！\s、]|やで$", asst))
    contam_total += len(re.findall(DESU_MASU, asst))
purity = osaka_total / max(osaka_total + contam_total, 1) * 100
print(f"  大阪弁ヒット: {osaka_total} ({osaka_total/n:.1f}/件)")
print(f"  汚染ヒット:   {contam_total} ({contam_total/n:.1f}/件)")
print(f"  純度: {purity:.1f}%")
