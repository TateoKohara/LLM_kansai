"""v4評価スクリプト: ベース / v2 / v4(iter1600) / v4(iter2000) を比較"""
import json
import math
import re
from collections import Counter

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

# ============================================================
#  パープレキシティ
# ============================================================

def compute_perplexity(model, tokenizer, test_file, max_seq=2048, label="model"):
    lines = open(test_file).readlines()
    total_loss = 0.0
    total_tokens = 0
    for i, line in enumerate(lines):
        data = json.loads(line)
        text = tokenizer.apply_chat_template(
            data["messages"], tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer.encode(text)
        if len(tokens) > max_seq:
            tokens = tokens[:max_seq]
        tokens_mx = mx.array(tokens)[None]
        logits = model(tokens_mx)
        targets = tokens_mx[:, 1:]
        loss = mx.mean(nn.losses.cross_entropy(logits[:, :-1, :], targets)).item()
        total_loss += loss * targets.size
        total_tokens += targets.size
        if (i + 1) % 50 == 0:
            print(f"  [{label}] {i+1}/{len(lines)}...")
        mx.eval()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss), len(lines)

# ============================================================
#  大阪弁品質分析
# ============================================================

OSAKA_PATTERNS = {
    "やで": r"やで[。！\s、]|やで$",
    "やねん": r"やねん[。！\s、]|やねん$",
    "やろ": r"やろ[。！？\s、]|やろ$",
    "やな": r"やな[。！\s、]|やな$",
    "やんか": r"やんか",
    "やわ": r"やわ[。！\s、]|やわ$",
    "やんな": r"やんな",
    "ねん": r"ねん[。！\s、]|ねん$",
    "めっちゃ": r"めっちゃ",
    "ほんま": r"ほんま",
    "あかん": r"あかん",
    "ちゃう": r"ちゃう",
    "おもろい": r"おもろい",
    "ええ": r"ええ[よやで。！]",
    "しとる": r"しとる|しとん",
    "せやな": r"せやな",
    "おる": r"おる[。！\s、やで]|おるん|おんねん",
    "へん": r"[あらさかたなまいきしちにひみりぎじびぴけせてねへめれ]へん",
    "でけへん": r"でけへん|できひん",
    "せなあかん": r"せなあかん|せなあかへん",
}

CONTAMINATION_PATTERNS = {
    "京都弁_どす": r"どす[。！\s、]|どす$",
    "京都弁_はる": r"してはる|いてはる|言うてはる",
    "標準語_です": r"です[。！\s、]|です$",
    "標準語_ます": r"ます[。！\s、]|ます$",
    "標準語_だよ": r"だよ[。！\s、ね]|だよ$",
    "標準語_だね": r"だね[。！\s、]|だね$",
}

PROMPTS = [
    "自己紹介をしてください。",
    "今日の天気はどうですか？",
    "おすすめの大阪グルメを教えてください。",
    "プログラミングを始めたいのですが、何から始めればいいですか？",
    "最近疲れています。元気が出るアドバイスをください。",
    "日本の四季の魅力について教えてください。",
    "子供に勉強する意味をどう説明しますか？",
    "友達と喧嘩してしまいました。どうすればいいですか？",
    "健康的な食事のコツを教えてください。",
    "人工知能について簡単に説明してください。",
    "読書の楽しさについて教えてください。",
    "朝起きるのが苦手です。コツはありますか？",
    "お金の貯め方を教えてください。",
    "ストレス解消法を教えてください。",
    "なぜ空は青いのですか？",
    "料理が上手になるコツは何ですか？",
    "英語を話せるようになりたいです。アドバイスをください。",
    "幸せとは何だと思いますか？",
    "運動を習慣にするにはどうしたらいいですか？",
    "1+1は何ですか？",
]


def evaluate_model(model, tokenizer, label):
    SYSTEM = "あんたは大阪弁で話す気さくなアシスタントやで。どんな質問にも大阪弁で答えてな。 /no_think"
    sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20)
    logits_proc = make_logits_processors(repetition_penalty=1.2, repetition_context_size=100)

    total_osaka = 0
    total_contam = 0
    osaka_counter = Counter()
    contam_counter = Counter()
    responses = []

    for i, prompt in enumerate(PROMPTS, 1):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=text, max_tokens=256,
                          sampler=sampler, logits_processors=logits_proc)
        clean = re.sub(r"<think>\s*</think>\s*", "", response).strip()

        o_hits = 0
        c_hits = 0
        for name, pat in OSAKA_PATTERNS.items():
            n = len(re.findall(pat, clean))
            if n:
                osaka_counter[name] += n
                o_hits += n
        for name, pat in CONTAMINATION_PATTERNS.items():
            n = len(re.findall(pat, clean))
            if n:
                contam_counter[name] += n
                c_hits += n

        total_osaka += o_hits
        total_contam += c_hits
        responses.append((i, prompt, clean, o_hits, c_hits))

    return total_osaka, total_contam, osaka_counter, contam_counter, responses


def print_results(label, total_osaka, total_contam, osaka_counter, contam_counter, responses):
    n = len(responses)
    purity = total_osaka / max(total_osaka + total_contam, 1) * 100
    avg_len = sum(len(r[2]) for r in responses) / n

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  大阪弁ヒット: {total_osaka} ({total_osaka/n:.1f}/件)")
    print(f"  汚染ヒット:   {total_contam} ({total_contam/n:.1f}/件)")
    print(f"  大阪弁純度:   {purity:.1f}%")
    print(f"  平均応答長:   {avg_len:.0f}字")

    print(f"\n  大阪弁パターン:")
    for k, v in osaka_counter.most_common(10):
        print(f"    {k:12s}: {v:3d}")

    if contam_counter:
        print(f"\n  汚染パターン:")
        for k, v in contam_counter.most_common():
            print(f"    {k:12s}: {v:3d}")

    print(f"\n  --- 個別回答(先頭80字) ---")
    for i, prompt, clean, oh, ch in responses:
        flag = " ***" if ch > 0 else ""
        preview = clean[:80].replace("\n", " ")
        print(f"  Q{i:2d}[O={oh},C={ch}{flag}]: {preview}")

    return {
        "label": label,
        "osaka_hits": total_osaka,
        "contam_hits": total_contam,
        "purity": purity,
        "avg_len": avg_len,
    }


def main():
    print("=" * 70)
    print("  v4 包括的評価: ベース / v2 / v4(1600) / v4(2000)")
    print("=" * 70)

    test_files = {
        "v3": "./data/osaka_data_v3/test.jsonl",
        "v4": "./data/osaka_data_v4/test.jsonl",
    }

    configs = [
        ("ベース", None),
        ("v2", "./adapters_v2_backup"),
        ("v4_iter1600", "./adapters_v4_1600"),
        ("v4_iter2000", "./adapters"),
    ]

    # Part 1: PPL
    print("\n" + "=" * 70)
    print("  Part 1: パープレキシティ")
    print("=" * 70)

    ppl_results = {}
    for label, adapter in configs:
        print(f"\n[{label}] 読み込み中...")
        if adapter is None:
            model, tok = load("./mlx_model")
        else:
            model, tok = load("./mlx_model", adapter_path=adapter)

        ppl_results[label] = {}
        for tname, tpath in test_files.items():
            loss, ppl, n = compute_perplexity(model, tok, tpath, label=f"{label}/{tname}")
            ppl_results[label][tname] = ppl
            print(f"  {label} on {tname}: PPL={ppl:.3f} (loss={loss:.3f}, n={n})")

        del model
        mx.eval()

    print(f"\n{'─'*60}")
    print(f"{'モデル':20s} {'v3 test PPL':>15s} {'v4 test PPL':>15s}")
    print(f"{'─'*60}")
    for label in ["ベース", "v2", "v4_iter1600", "v4_iter2000"]:
        v3p = ppl_results[label].get("v3", 0)
        v4p = ppl_results[label].get("v4", 0)
        print(f"{label:20s} {v3p:15.3f} {v4p:15.3f}")

    # Part 2: 定性評価
    print("\n" + "=" * 70)
    print("  Part 2: 定性評価 (20問)")
    print("=" * 70)

    summaries = []
    for label, adapter in configs:
        print(f"\n[{label}] 読み込み中...")
        if adapter is None:
            model, tok = load("./mlx_model")
        else:
            model, tok = load("./mlx_model", adapter_path=adapter)

        to, tc, oc, cc, resp = evaluate_model(model, tok, label)
        s = print_results(label, to, tc, oc, cc, resp)
        summaries.append(s)

        del model
        mx.eval()

    # Final summary
    print(f"\n{'='*70}")
    print(f"  最終比較サマリ")
    print(f"{'='*70}")
    print(f"{'指標':20s}", end="")
    for s in summaries:
        print(f" {s['label']:>15s}", end="")
    print()
    print(f"{'─'*80}")

    print(f"{'大阪弁ヒット':20s}", end="")
    for s in summaries:
        print(f" {s['osaka_hits']:>15d}", end="")
    print()

    print(f"{'汚染ヒット':20s}", end="")
    for s in summaries:
        print(f" {s['contam_hits']:>15d}", end="")
    print()

    print(f"{'大阪弁純度(%)':20s}", end="")
    for s in summaries:
        print(f" {s['purity']:>15.1f}", end="")
    print()

    print(f"{'平均応答長(字)':20s}", end="")
    for s in summaries:
        print(f" {s['avg_len']:>15.0f}", end="")
    print()

    # PPL
    print(f"{'PPL (v4 test)':20s}", end="")
    for s in summaries:
        p = ppl_results.get(s["label"], {}).get("v4", 0)
        print(f" {p:>15.1f}", end="")
    print()


if __name__ == "__main__":
    main()
