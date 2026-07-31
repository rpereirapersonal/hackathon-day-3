#!/usr/bin/env python3
"""Small, time-boxed LoRA fine-tune of Llama-3.1-Nemotron-Nano-8B-v1.

Uses the reference baseline hyperparameters from Setup_Instructions.md /
01_training_guide.md (LoRA rank 32, lr 5e-5, seq_len 512) but caps max_steps
low so a full run finishes inside a short demo window instead of the
documented ~2-3 hour full run.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def build_text(example, tokenizer):
    return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_steps", type=int, default=5)
    ap.add_argument("--checkpoint_every", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[{0:.1f}s] Loading tokenizer/model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    print(f"[{time.time()-t0:.1f}s] Model loaded")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw_ds = load_dataset("json", data_files={"train": args.train_file, "validation": args.val_file})

    def tokenize_fn(example):
        text = build_text(example, tokenizer)
        toks = tokenizer(text, truncation=True, max_length=args.max_seq_len, padding="max_length")
        toks["labels"] = toks["input_ids"].copy()
        return toks

    tokenized = raw_ds.map(tokenize_fn, remove_columns=raw_ds["train"].column_names)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "hf_run"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=args.checkpoint_every,
        save_total_limit=5,
        eval_strategy="steps",
        eval_steps=args.checkpoint_every,
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        logging_dir=str(out_dir / "hf_run" / "tb_logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
    )

    print(f"[{time.time()-t0:.1f}s] Starting training: {args.max_steps} steps")
    train_result = trainer.train()
    print(f"[{time.time()-t0:.1f}s] Training complete")

    metrics_path = out_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    eval_metrics = trainer.evaluate()
    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print("Eval metrics:", eval_metrics)

    adapter_dir = out_dir / "hf_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[{time.time()-t0:.1f}s] Adapter saved to {adapter_dir}")

    with open(out_dir / "log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)


if __name__ == "__main__":
    main()
