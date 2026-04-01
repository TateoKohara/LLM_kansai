#!/usr/bin/env python3
"""
Step 1.2: ITA_KANSAI_CORPUS分析・大阪弁パターン辞書構築
- joumonsugi/ITA_KANSAI_CORPUS からパターン抽出
- 語尾・語彙・感情表現パターンの辞書構築
- 京都弁・神戸弁除外ガイドライン作成
- data/osaka_patterns.json (構造化辞書)
- data/osaka_style_guide.md (LLMプロンプト用スタイルガイド)
"""

import json
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path(__file__).parent.parent / "data"
CORPUS_DIR = DATA_DIR / "ita_kansai_corpus"
PATTERNS_FILE = DATA_DIR / "osaka_patterns.json"
STYLE_GUIDE_FILE = DATA_DIR / "osaka_style_guide.md"

CORPUS_ZIP_URL = "https://github.com/joumonsugi/ITA_KANSAI_CORPUS/archive/refs/heads/main.zip"


# ---------------------------------------------------------------------------
# 言語学的知識に基づく大阪弁パターン定義
# ---------------------------------------------------------------------------

# 語尾変換ルール (標準語 → 大阪弁)
WORD_ENDINGS = [
    # コピュラ・断定
    {"standard": "です", "osaka": "や", "context": "断定"},
    {"standard": "ですね", "osaka": "やね", "context": "同意求め"},
    {"standard": "ですよ", "osaka": "やで", "context": "強調・主張"},
    {"standard": "ですか", "osaka": "やんか", "context": "確認"},
    {"standard": "でしょう", "osaka": "やろ", "context": "推量"},
    {"standard": "でしょ？", "osaka": "やろ？", "context": "確認"},
    {"standard": "ではない", "osaka": "ちゃう", "context": "否定"},
    {"standard": "じゃない", "osaka": "ちゃう", "context": "否定"},
    {"standard": "ません", "osaka": "へん", "context": "丁寧否定"},
    {"standard": "ない", "osaka": "へん", "context": "否定"},
    # 動詞語尾
    {"standard": "している", "osaka": "してる・しとる", "context": "進行"},
    {"standard": "しています", "osaka": "してんねん", "context": "進行・説明"},
    {"standard": "しました", "osaka": "してん", "context": "過去"},
    {"standard": "してください", "osaka": "してんか・してや", "context": "依頼"},
    {"standard": "しなければ", "osaka": "せなあかん", "context": "義務"},
    {"standard": "できない", "osaka": "でけへん・できひん", "context": "不可能"},
    {"standard": "いけない", "osaka": "あかん", "context": "禁止"},
    # 理由
    {"standard": "なんです", "osaka": "やねん", "context": "理由・説明"},
    {"standard": "だから", "osaka": "やから", "context": "理由"},
    {"standard": "ですから", "osaka": "やさかい", "context": "理由（やや古め）"},
    # 終助詞
    {"standard": "ね", "osaka": "な", "context": "同意"},
    {"standard": "よね", "osaka": "やんな", "context": "確認"},
    {"standard": "だよ", "osaka": "やで", "context": "主張"},
    {"standard": "のです", "osaka": "ねん", "context": "説明"},
]

# 語彙変換 (標準語 → 大阪弁)
VOCABULARY = [
    {"standard": "とても・すごく", "osaka": "めっちゃ", "freq": "高"},
    {"standard": "本当に", "osaka": "ほんまに", "freq": "高"},
    {"standard": "本当", "osaka": "ほんま", "freq": "高"},
    {"standard": "違う", "osaka": "ちゃう", "freq": "高"},
    {"standard": "面白い", "osaka": "おもろい", "freq": "高"},
    {"standard": "ダメ・いけない", "osaka": "あかん", "freq": "高"},
    {"standard": "なぜ・どうして", "osaka": "なんで", "freq": "高"},
    {"standard": "自分", "osaka": "自分・わて（古め）", "freq": "中"},
    {"standard": "あなた", "osaka": "あんた・自分", "freq": "中"},
    {"standard": "良い", "osaka": "ええ", "freq": "高"},
    {"standard": "つまらない", "osaka": "おもんない", "freq": "中"},
    {"standard": "捨てる", "osaka": "ほかす", "freq": "中"},
    {"standard": "片付ける", "osaka": "なおす", "freq": "中"},
    {"standard": "突っ込む", "osaka": "つっこむ", "freq": "中"},
    {"standard": "知らない", "osaka": "知らんわ", "freq": "高"},
    {"standard": "すみません", "osaka": "すんません・すまんな", "freq": "高"},
    {"standard": "ありがとう", "osaka": "おおきに", "freq": "中"},
    {"standard": "たくさん", "osaka": "ぎょうさん", "freq": "中"},
    {"standard": "大丈夫", "osaka": "だいじょぶやで", "freq": "高"},
    {"standard": "びっくりする", "osaka": "びっくりする・たまげる", "freq": "中"},
    {"standard": "怒る", "osaka": "怒る・いかる", "freq": "中"},
    {"standard": "しんどい（疲れる）", "osaka": "しんどい", "freq": "高"},
    {"standard": "無理", "osaka": "無理やわ", "freq": "高"},
]

# 感情表現パターン
EMOTION_PATTERNS = {
    "surprise": [
        "えーっ！", "マジで！？", "ほんまに！？", "なんでやねん！",
        "うそやん！", "びっくりしたわ！",
    ],
    "anger": [
        "なんやねん！", "いい加減にせえや！", "あほちゃう？",
        "ふざけんなや！", "しばくぞ！",
    ],
    "laugh": [
        "おもろいやん！", "ウケるわ〜", "めっちゃ笑えるわ",
        "あはは、ええやん", "さすがやな〜",
    ],
    "sad": [
        "かなんわ…", "しゃあないな…", "つらいわ〜",
        "あかんわ…", "悲しいなぁ",
    ],
    "agreement": [
        "せやな", "せやせや", "ほんまそれ", "そうやんな",
        "わかるわ〜", "ええと思うで",
    ],
}

# 特徴的なフレーズ
CHARACTERISTIC_PHRASES = [
    "なんでやねん",
    "知らんけど",
    "ほな",
    "まあええわ",
    "そないなことあるかいな",
    "せやかて",
    "よう言うわ",
    "あんたなぁ",
    "そらそうや",
    "かめへんかめへん",
]

# 京都弁・神戸弁の除外パターン
EXCLUDE_PATTERNS = {
    "kyoto": {
        "description": "京都弁 — 以下は使用しない",
        "patterns": [
            "〜どす", "〜どすえ", "〜え（語尾）",
            "〜はる（尊敬）", "おこしやす", "おいでやす",
            "〜しはる", "よろしおす", "堪忍え",
        ],
    },
    "kobe": {
        "description": "神戸弁 — 以下は使用しない",
        "patterns": [
            "〜とう（しとう＝している）", "〜しとん？",
            "〜やっとう", "〜しとって", "なんしよん？",
        ],
    },
}


# ---------------------------------------------------------------------------
# ITA_KANSAI_CORPUS からの抽出
# ---------------------------------------------------------------------------

def download_ita_corpus() -> Path | None:
    """GitHubからITA_KANSAI_CORPUSをダウンロード・展開"""
    print("📦 Downloading ITA_KANSAI_CORPUS from GitHub ...")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    # すでに展開済みか確認
    existing = list(CORPUS_DIR.glob("**/*.txt"))
    if existing:
        print(f"  Already extracted ({len(existing)} txt files)")
        return CORPUS_DIR

    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            urlretrieve(CORPUS_ZIP_URL, tmp.name)
            print(f"  Downloaded to {tmp.name}")

        with zipfile.ZipFile(tmp.name, "r") as zf:
            zf.extractall(CORPUS_DIR)
            print(f"  Extracted {len(zf.namelist())} files")

        Path(tmp.name).unlink()
        return CORPUS_DIR
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return None


def analyze_ita_corpus() -> dict:
    """ITA_KANSAI_CORPUS のテキストファイルを分析してパターンを抽出"""
    corpus_stats: dict = {"total_entries": 0, "texts": [], "marker_frequencies": {}}

    corpus_dir = download_ita_corpus()
    if not corpus_dir:
        return corpus_stats

    # テキストファイルを読み込み
    txt_files = sorted(corpus_dir.glob("**/*.txt"))
    print(f"\n📄 Found {len(txt_files)} text files:")

    all_texts: list[str] = []
    emotion_texts: dict[str, list[str]] = {
        "ANGER": [], "LAUGH": [], "SAD": [], "SURPRISE": [],
    }

    for fp in txt_files:
        try:
            content = fp.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = fp.read_text(encoding="shift_jis")
            except Exception:
                continue

        lines = [l.strip() for l in content.splitlines() if l.strip()]
        rel = fp.relative_to(corpus_dir)
        print(f"  {rel}: {len(lines)} lines")

        # 感情差分リストの振り分け
        name_upper = fp.stem.upper()
        for emo in emotion_texts:
            if emo in name_upper:
                emotion_texts[emo].extend(lines)

        all_texts.extend(lines)

    corpus_stats["total_entries"] = len(all_texts)
    corpus_stats["texts"] = all_texts[:50]  # サンプル保存
    print(f"\n  Total text lines: {len(all_texts)}")

    if not all_texts:
        return corpus_stats

    # サンプル表示
    print("\n  Sample lines:")
    for line in all_texts[:10]:
        print(f"    {line}")

    # 大阪弁マーカー頻度分析
    osaka_markers = [
        "やで", "やねん", "やろ", "やんか", "やな", "やけど",
        "ねん", "へん", "てん", "めっちゃ", "ほんま", "あかん",
        "ちゃう", "おもろ", "せや", "しとる", "しとん",
        "やんな", "やわ", "やん", "やから", "ええ",
        "なんで", "あんた", "ほな", "知らんけど",
    ]
    ending_patterns = Counter()
    for text in all_texts:
        for marker in osaka_markers:
            if marker in text:
                ending_patterns[marker] += 1

    corpus_stats["marker_frequencies"] = dict(ending_patterns.most_common())
    print("\n  Osaka dialect markers found in corpus:")
    for marker, freq in ending_patterns.most_common(25):
        print(f"    {marker}: {freq}")

    # 感情別テキスト統計
    corpus_stats["emotion_texts"] = {}
    for emo, texts in emotion_texts.items():
        if texts:
            corpus_stats["emotion_texts"][emo] = texts
            print(f"\n  Emotion [{emo}]: {len(texts)} lines")
            for t in texts[:3]:
                print(f"    {t}")

    return corpus_stats


# ---------------------------------------------------------------------------
# パターン辞書 JSON 生成
# ---------------------------------------------------------------------------

def build_pattern_dict(corpus_stats: dict) -> dict:
    """構造化パターン辞書を生成"""
    pattern_dict = {
        "metadata": {
            "description": "大阪弁変換パターン辞書",
            "scope": "大阪弁に限定（京都弁・神戸弁は除外）",
            "source": "言語学的知識 + ITA_KANSAI_CORPUS分析",
        },
        "word_endings": WORD_ENDINGS,
        "vocabulary": VOCABULARY,
        "emotion_patterns": EMOTION_PATTERNS,
        "characteristic_phrases": CHARACTERISTIC_PHRASES,
        "exclude_patterns": EXCLUDE_PATTERNS,
    }
    if corpus_stats.get("marker_frequencies"):
        pattern_dict["corpus_marker_frequencies"] = corpus_stats["marker_frequencies"]
    return pattern_dict


# ---------------------------------------------------------------------------
# LLMプロンプト用スタイルガイド生成
# ---------------------------------------------------------------------------

def build_style_guide(pattern_dict: dict) -> str:
    """Step 1.3で使う大阪弁変換プロンプト用スタイルガイド"""
    lines = [
        "# 大阪弁変換スタイルガイド",
        "",
        "## 基本方針",
        "- **大阪弁のみ**で回答する（京都弁・神戸弁は使わない）",
        "- 自然でくだけた口調を心がける",
        "- ユーザーの質問（標準語）はそのまま、アシスタントの応答のみ大阪弁にする",
        "",
        "## 語尾変換ルール",
        "",
        "| 標準語 | 大阪弁 | 場面 |",
        "|---|---|---|",
    ]
    for e in pattern_dict["word_endings"]:
        lines.append(f"| {e['standard']} | {e['osaka']} | {e['context']} |")

    lines += [
        "",
        "## 語彙変換",
        "",
        "| 標準語 | 大阪弁 | 頻度 |",
        "|---|---|---|",
    ]
    for v in pattern_dict["vocabulary"]:
        lines.append(f"| {v['standard']} | {v['osaka']} | {v['freq']} |")

    lines += [
        "",
        "## 感情表現",
        "",
    ]
    for emotion, phrases in pattern_dict["emotion_patterns"].items():
        lines.append(f"### {emotion}")
        for p in phrases:
            lines.append(f"- {p}")
        lines.append("")

    lines += [
        "## 特徴的なフレーズ",
        "",
    ]
    for p in pattern_dict["characteristic_phrases"]:
        lines.append(f"- {p}")

    lines += [
        "",
        "## ⚠️ 除外パターン（使用禁止）",
        "",
    ]
    for dialect, info in pattern_dict["exclude_patterns"].items():
        lines.append(f"### {info['description']}")
        for p in info["patterns"]:
            lines.append(f"- ❌ {p}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- ITA_KANSAI_CORPUS 分析 ---
    corpus_stats = analyze_ita_corpus()

    # --- パターン辞書構築 ---
    print("\n📝 Building pattern dictionary ...")
    pattern_dict = build_pattern_dict(corpus_stats)

    with open(PATTERNS_FILE, "w", encoding="utf-8") as f:
        json.dump(pattern_dict, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved to {PATTERNS_FILE}")

    # --- スタイルガイド生成 ---
    print("📝 Building style guide for LLM prompts ...")
    guide = build_style_guide(pattern_dict)

    with open(STYLE_GUIDE_FILE, "w", encoding="utf-8") as f:
        f.write(guide)
    print(f"💾 Saved to {STYLE_GUIDE_FILE}")

    # --- サマリー ---
    print(f"\n{'='*50}")
    print("Pattern dictionary summary:")
    print(f"  Word endings:     {len(pattern_dict['word_endings'])} rules")
    print(f"  Vocabulary:       {len(pattern_dict['vocabulary'])} entries")
    print(f"  Emotions:         {len(pattern_dict['emotion_patterns'])} categories")
    print(f"  Phrases:          {len(pattern_dict['characteristic_phrases'])} entries")
    print(f"  Exclude patterns: {sum(len(v['patterns']) for v in pattern_dict['exclude_patterns'].values())} entries")
    if corpus_stats.get("marker_frequencies"):
        print(f"  Corpus markers:   {len(corpus_stats['marker_frequencies'])} found")


if __name__ == "__main__":
    main()
