#!/usr/bin/env python3
"""Generate base-model and fine-tuned-model answers on the same held-out
prompts so the two can be compared side by side as training evidence."""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompts(val_file):
    prompts = []
    with open(val_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            msgs = ex["messages"]
            system, user, gold = msgs[0]["content"], msgs[1]["content"], msgs[2]["content"]
            prompts.append({"system": system, "user": user, "gold": gold})
    return prompts


def generate(model, tokenizer, system, user, max_new_tokens=200):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--adapter_path", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    prompts = load_prompts(args.val_file)[: args.n]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    base_model.eval()

    results = []
    print(f"Generating base-model answers for {len(prompts)} prompts...")
    for i, p in enumerate(prompts):
        ans = generate(base_model, tokenizer, p["system"], p["user"])
        results.append({"question": p["user"], "gold": p["gold"], "base_answer": ans})
        print(f"  [{i+1}/{len(prompts)}] done")

    print("Attaching LoRA adapter...")
    ft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    ft_model.eval()

    print(f"Generating fine-tuned-model answers for {len(prompts)} prompts...")
    for i, r in enumerate(results):
        p = prompts[i]
        ans = generate(ft_model, tokenizer, p["system"], p["user"])
        r["finetuned_answer"] = ans
        print(f"  [{i+1}/{len(prompts)}] done")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote comparison to {out_path}")


if __name__ == "__main__":
    main()
