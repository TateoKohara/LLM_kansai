#!/usr/bin/env python3
"""大阪弁LLM 対話スクリプト（repetition_penalty対応）"""

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

MODEL_PATH = "./mlx_model"
ADAPTER_PATH = "./adapters"
SYSTEM_PROMPT = "あんたは大阪弁で話す気さくなアシスタントやで。どんな質問にも大阪弁で答えてな。 /no_think"

print("モデル読み込み中...")
model, tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)
sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20)
logits_proc = make_logits_processors(repetition_penalty=1.2, repetition_context_size=100)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print("大阪弁チャット起動したで！ 'q' で終了、'r' でリセットや。\n")

while True:
    try:
        user_input = input("あんた>> ")
    except (EOFError, KeyboardInterrupt):
        print("\nほな、さいなら！")
        break

    if user_input.strip().lower() == "q":
        print("ほな、さいなら！")
        break
    if user_input.strip().lower() == "r":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        print("[会話リセットしたで]\n")
        continue
    if not user_input.strip():
        continue

    messages.append({"role": "user", "content": user_input})
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        sampler=sampler,
        logits_processors=logits_proc,
        max_tokens=512,
        verbose=False,
    )

    # /no_think で思考抑制しているが、念のため <think> タグを除去
    clean = response
    if "<think>" in clean:
        import re
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()

    print(f"大阪弁AI>> {clean}\n")
    messages.append({"role": "assistant", "content": clean})
