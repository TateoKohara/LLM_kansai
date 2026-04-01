#!/usr/bin/env python3
"""
Step 1.3: 大阪弁変換パイプライン
- GPT-4o / Claude 各50件ずつ比較テスト (--mode compare)
- 本番バッチ変換 (--mode batch --api <openai|anthropic>)
- asyncio + rate limiting で効率的にAPI呼び出し
"""

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv

# .env ファイルから環境変数を読み込み
load_dotenv(Path(__file__).parent.parent / ".env")

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_FILE = DATA_DIR / "raw_conversations.jsonl"
STYLE_GUIDE_FILE = DATA_DIR / "osaka_style_guide.md"
COMPARE_DIR = DATA_DIR / "compare_test"
OUTPUT_FILE = DATA_DIR / "osaka_conversations.jsonl"

SEED = 42
COMPARE_COUNT = 50

# --- Rate limiting ---
# OpenAI: Tier 1 = 500 RPM → ~8 RPS, 余裕を見て 5 RPS
# Anthropic: Tier 1 = 50 RPM → ~0.8 RPS, 余裕を見て 0.5 RPS
RATE_LIMITS = {
    "openai": 5.0,
    "anthropic": 0.5,
}


def load_style_guide() -> str:
    return STYLE_GUIDE_FILE.read_text(encoding="utf-8")


def build_system_prompt(style_guide: str) -> str:
    return f"""あなたは「標準語の日本語テキストを大阪弁に変換する」専門の翻訳者です。

## タスク
以下の「アシスタントの応答」を大阪弁に変換してください。

## ルール
1. **大阪弁のみ**を使用すること（京都弁・神戸弁は絶対に使わない）
2. 内容・情報の正確性はそのまま保つこと
3. 自然な大阪弁の会話体にすること（過度に誇張しない）
4. 固有名詞・数値・専門用語はそのまま残すこと
5. リスト形式の応答はリスト形式を維持すること
6. 京都弁（〜どす、〜はる）や神戸弁（〜とう、〜しとん）は使わないこと

## 大阪弁スタイルガイド
{style_guide}

## 出力形式
変換後の大阪弁テキストのみを出力してください。説明や注釈は不要です。"""


def build_user_prompt(instruction: str, response: str) -> str:
    return f"""## ユーザーの質問（参考・変換不要）
{instruction}

## アシスタントの応答（これを大阪弁に変換してください）
{response}"""


# ---------------------------------------------------------------------------
# API clients (async)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token bucket rate limiter"""
    def __init__(self, rate: float):
        self.rate = rate
        self.interval = 1.0 / rate
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self._last + self.interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


async def call_openai(
    client,
    system_prompt: str,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
    limiter: RateLimiter,
) -> str:
    await limiter.acquire()
    async with semaphore:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        return response.choices[0].message.content.strip()


async def call_anthropic(
    client,
    system_prompt: str,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
    limiter: RateLimiter,
) -> str:
    await limiter.acquire()
    async with semaphore:
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# 比較テスト
# ---------------------------------------------------------------------------

async def run_compare_test():
    """GPT-4o / Claude 各50件で変換テスト"""

    # 両方のAPIキーが必要
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openai_key:
        print("❌ OPENAI_API_KEY が設定されていません")
        return
    if not anthropic_key:
        print("❌ ANTHROPIC_API_KEY が設定されていません")
        return

    from anthropic import AsyncAnthropic
    from openai import AsyncOpenAI

    openai_client = AsyncOpenAI(api_key=openai_key)
    anthropic_client = AsyncAnthropic(api_key=anthropic_key)

    # データ読み込み・サンプリング
    with open(RAW_FILE, encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]

    random.seed(SEED)
    samples = random.sample(all_data, min(COMPARE_COUNT, len(all_data)))

    style_guide = load_style_guide()
    system_prompt = build_system_prompt(style_guide)

    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🔄 比較テスト開始: {len(samples)} 件 × 2 API")
    print(f"   OpenAI rate: {RATE_LIMITS['openai']} RPS")
    print(f"   Anthropic rate: {RATE_LIMITS['anthropic']} RPS")

    openai_limiter = RateLimiter(RATE_LIMITS["openai"])
    anthropic_limiter = RateLimiter(RATE_LIMITS["anthropic"])
    openai_sem = asyncio.Semaphore(10)
    anthropic_sem = asyncio.Semaphore(5)

    results = []
    openai_cost_input = 0
    openai_cost_output = 0
    anthropic_cost_input = 0
    anthropic_cost_output = 0

    for i, sample in enumerate(samples):
        user_prompt = build_user_prompt(sample["instruction"], sample["response"])
        print(f"\n  [{i+1}/{len(samples)}] {sample['instruction'][:50]}...")

        # OpenAI
        t0 = time.time()
        try:
            openai_result = await call_openai(
                openai_client, system_prompt, user_prompt, openai_sem, openai_limiter,
            )
            openai_time = time.time() - t0
            print(f"    GPT-4o: {openai_result[:60]}... ({openai_time:.1f}s)")
        except Exception as e:
            openai_result = f"[ERROR] {e}"
            openai_time = time.time() - t0
            print(f"    GPT-4o: ERROR - {e}")

        # Anthropic
        t0 = time.time()
        try:
            anthropic_result = await call_anthropic(
                anthropic_client, system_prompt, user_prompt, anthropic_sem, anthropic_limiter,
            )
            anthropic_time = time.time() - t0
            print(f"    Claude: {anthropic_result[:60]}... ({anthropic_time:.1f}s)")
        except Exception as e:
            anthropic_result = f"[ERROR] {e}"
            anthropic_time = time.time() - t0
            print(f"    Claude: ERROR - {e}")

        results.append({
            "index": i,
            "instruction": sample["instruction"],
            "original_response": sample["response"],
            "source": sample["source"],
            "gpt4o_response": openai_result,
            "gpt4o_time": openai_time,
            "claude_response": anthropic_result,
            "claude_time": anthropic_time,
        })

    # 結果を保存
    compare_file = COMPARE_DIR / "compare_results.jsonl"
    with open(compare_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # サマリー
    gpt_errors = sum(1 for r in results if r["gpt4o_response"].startswith("[ERROR]"))
    claude_errors = sum(1 for r in results if r["claude_response"].startswith("[ERROR]"))
    gpt_avg_time = sum(r["gpt4o_time"] for r in results) / len(results)
    claude_avg_time = sum(r["claude_time"] for r in results) / len(results)

    summary = {
        "total_samples": len(results),
        "gpt4o": {
            "errors": gpt_errors,
            "success": len(results) - gpt_errors,
            "avg_time_sec": round(gpt_avg_time, 2),
        },
        "claude": {
            "errors": claude_errors,
            "success": len(results) - claude_errors,
            "avg_time_sec": round(claude_avg_time, 2),
        },
    }

    summary_file = COMPARE_DIR / "compare_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 人手レビュー用のMarkdownも生成
    review_file = COMPARE_DIR / "compare_review.md"
    with open(review_file, "w", encoding="utf-8") as f:
        f.write("# 大阪弁変換 比較テスト結果\n\n")
        f.write(f"サンプル数: {len(results)}件\n\n")
        f.write(f"| 指標 | GPT-4o | Claude |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| 成功 | {summary['gpt4o']['success']} | {summary['claude']['success']} |\n")
        f.write(f"| エラー | {summary['gpt4o']['errors']} | {summary['claude']['errors']} |\n")
        f.write(f"| 平均応答時間 | {summary['gpt4o']['avg_time_sec']}s | {summary['claude']['avg_time_sec']}s |\n")
        f.write("\n---\n\n")

        for r in results[:20]:  # 最初の20件をレビュー用に表示
            f.write(f"## Sample {r['index']+1}\n\n")
            f.write(f"**質問:** {r['instruction'][:200]}\n\n")
            f.write(f"**元の応答:** {r['original_response'][:300]}\n\n")
            f.write(f"**GPT-4o:** {r['gpt4o_response'][:300]}\n\n")
            f.write(f"**Claude:** {r['claude_response'][:300]}\n\n")
            f.write("---\n\n")

    print(f"\n{'='*50}")
    print(f"比較テスト完了!")
    print(f"  結果: {compare_file}")
    print(f"  サマリー: {summary_file}")
    print(f"  レビュー: {review_file}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# 本番バッチ変換
# ---------------------------------------------------------------------------

async def run_batch(api: str, max_concurrent: int = 10):
    """全5,000件を指定APIで大阪弁変換"""

    if api == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("❌ OPENAI_API_KEY が設定されていません")
            return
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=key)
    elif api == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("❌ ANTHROPIC_API_KEY が設定されていません")
            return
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=key)
    else:
        print(f"❌ Unknown API: {api}")
        return

    # データ読み込み
    with open(RAW_FILE, encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]

    # 既に処理済みのものをスキップ（リジューム対応）
    done_indices: set[int] = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                done_indices.add(row["index"])
        print(f"📂 リジューム: {len(done_indices)} 件処理済み")

    remaining = [(i, d) for i, d in enumerate(all_data) if i not in done_indices]
    if not remaining:
        print("✅ 全件処理済みです")
        return

    style_guide = load_style_guide()
    system_prompt = build_system_prompt(style_guide)
    limiter = RateLimiter(RATE_LIMITS[api])
    sem = asyncio.Semaphore(max_concurrent)

    print(f"🔄 バッチ変換開始: {len(remaining)} 件 (API: {api})")
    print(f"   Rate: {RATE_LIMITS[api]} RPS, Concurrency: {max_concurrent}")

    success = 0
    errors = 0
    start_time = time.time()

    # チャンク単位で処理してファイルに逐次書き込み
    CHUNK_SIZE = 50

    for chunk_start in range(0, len(remaining), CHUNK_SIZE):
        chunk = remaining[chunk_start : chunk_start + CHUNK_SIZE]

        async def process_one(idx: int, sample: dict) -> dict | None:
            user_prompt = build_user_prompt(sample["instruction"], sample["response"])
            try:
                if api == "openai":
                    result = await call_openai(client, system_prompt, user_prompt, sem, limiter)
                else:
                    result = await call_anthropic(client, system_prompt, user_prompt, sem, limiter)
                return {
                    "index": idx,
                    "instruction": sample["instruction"],
                    "original_response": sample["response"],
                    "osaka_response": result,
                    "source": sample["source"],
                    "api": api,
                }
            except Exception as e:
                print(f"  ❌ [{idx}] Error: {e}")
                return None

        tasks = [process_one(idx, sample) for idx, sample in chunk]
        chunk_results = await asyncio.gather(*tasks)

        # ファイルに追記
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for r in chunk_results:
                if r is not None:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    success += 1
                else:
                    errors += 1

        elapsed = time.time() - start_time
        total_done = chunk_start + len(chunk)
        rate = total_done / max(elapsed, 1)
        eta = (len(remaining) - total_done) / max(rate, 0.01)
        print(
            f"  進捗: {total_done}/{len(remaining)} "
            f"(成功: {success}, エラー: {errors}, "
            f"{rate:.1f} 件/s, ETA: {eta/60:.1f}分)"
        )

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"バッチ変換完了!")
    print(f"  成功: {success}, エラー: {errors}")
    print(f"  所要時間: {elapsed/60:.1f}分")
    print(f"  出力: {OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="大阪弁変換パイプライン")
    parser.add_argument(
        "--mode",
        choices=["compare", "batch"],
        required=True,
        help="compare: GPT-4o/Claude 50件比較テスト, batch: 全件変換",
    )
    parser.add_argument(
        "--api",
        choices=["openai", "anthropic"],
        default=None,
        help="batch モードで使用する API (比較テスト後に決定)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="最大同時リクエスト数 (default: 10)",
    )
    args = parser.parse_args()

    if args.mode == "compare":
        asyncio.run(run_compare_test())
    elif args.mode == "batch":
        if not args.api:
            print("❌ --api を指定してください (openai or anthropic)")
            return
        asyncio.run(run_batch(args.api, args.max_concurrent))


if __name__ == "__main__":
    main()
