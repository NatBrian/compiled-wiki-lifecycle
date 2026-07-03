"""Trained-maintainer stage: LoRA SFT of the maintainer (Qwen2.5-14B-Instruct by
default).

Single-GPU: set CUDA_VISIBLE_DEVICES. TRL SFTTrainer on a results/sft_*.jsonl (messages
format). Saves adapter to results/<out_name>/. Base defaults to 14B so the adapter serves
on the SAME base as the compiler/judge (consistent with the base model used throughout
the rest of this repo, including this stage's earlier diagnosis work).
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    torch.backends.cuda.enable_cudnn_sdp(False)   # box cuDNN workaround (also used in the
                                                   # certify stage and this stage's earlier diagnosis work)
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset("json", data_files=args.data, split="train").shuffle(seed=args.seed)
    n_eval = max(8, int(0.05 * len(ds)))
    eval_ds, train_ds = ds.select(range(n_eval)), ds.select(range(n_eval, len(ds)))
    print(f"base={args.base} train={len(train_ds)} eval={len(eval_ds)}")

    peft_cfg = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"],
                          task_type="CAUSAL_LM")
    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=5, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, bf16=True, max_length=args.max_len,
        gradient_checkpointing=True, report_to=[], seed=args.seed,
        completion_only_loss=True)
    trainer = SFTTrainer(model=args.base, args=cfg, train_dataset=train_ds,
                         eval_dataset=eval_ds, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.out)
    with open(os.path.join(args.out, "train_meta.json"), "w") as f:
        json.dump({"base": args.base, "n_train": len(train_ds), "epochs": args.epochs,
                   "rank": args.rank, "lr": args.lr, "data": args.data,
                   "final": trainer.state.log_history[-1] if trainer.state.log_history else None},
                  f, indent=2, default=str)
    print(f"adapter saved -> {args.out}")


if __name__ == "__main__":
    main()
