import argparse

import unsloth  # noqa: F401
import torch
from unsloth import FastModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.adapter_dir,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    FastModel.for_inference(model)

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a calm survival advisor. Prioritize safety, water, "
                        "shelter, signaling, first aid, and uncertainty-aware decision making."
                    ),
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": args.prompt}]},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    print(tokenizer.decode(generated_tokens, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
