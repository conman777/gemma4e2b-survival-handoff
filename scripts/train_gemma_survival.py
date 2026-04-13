import argparse
from pathlib import Path

import unsloth  # noqa: F401
import torch
from datasets import load_dataset
from transformers import set_seed
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="google/gemma-4-E2B-it")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument(
        "--evaluation-strategy",
        choices=("no", "steps", "epoch"),
        default="no",
    )
    parser.add_argument(
        "--save-strategy",
        choices=("steps", "epoch"),
        default="epoch",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def to_text(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_path = Path(args.dataset_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    raw = load_dataset("json", data_files=str(dataset_path), split="train")
    train_source = raw
    eval_source = None
    if args.evaluation_strategy != "no":
        test_size = 1 if len(raw) <= 10 else max(1, int(len(raw) * 0.1))
        split = raw.train_test_split(test_size=test_size, seed=args.seed)
        train_source = split["train"]
        eval_source = split["test"]

    train_dataset = train_source.map(
        to_text,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=train_source.column_names,
        num_proc=1,
        desc="Formatting train split",
    )
    eval_dataset = None
    if eval_source is not None:
        eval_dataset = eval_source.map(
            to_text,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=eval_source.column_names,
            num_proc=1,
            desc="Formatting eval split",
        )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    config_kwargs = dict(
        output_dir=str(output_dir),
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.evaluation_strategy,
        save_total_limit=2,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to="none",
        seed=args.seed,
    )
    if args.save_strategy == "steps":
        config_kwargs["save_steps"] = args.save_steps
    if args.evaluation_strategy == "steps":
        config_kwargs["eval_steps"] = args.eval_steps

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        dataset_num_proc=1,
        packing=False,
        args=SFTConfig(**config_kwargs),
    )

    trainer.train()
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved adapter and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
