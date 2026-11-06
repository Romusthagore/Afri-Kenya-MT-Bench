from pathlib import Path

import torch
import yaml
from peft import PeftModel

from src.models.load_model import load_model_and_tokenizer


def load_translation_model(config_path, adapter_path=None):
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model_name"],
        local_files_only=config["local_files_only"],
        torch_dtype=config["torch_dtype"],
    )

    if adapter_path is not None:
        adapter_path = Path(adapter_path)

        if not adapter_path.is_dir():
            raise FileNotFoundError(adapter_path)

        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )

    model = model.to("cuda")
    model.eval()

    return model, tokenizer


@torch.inference_mode()
def generate_translation(
    model,
    tokenizer,
    instruction,
    source_text,
    max_new_tokens=128,
):
    user_text = f"{instruction}\n\n{source_text}"

    is_qwen = "qwen" in tokenizer.name_or_path.lower()
    template_kwargs = {"enable_thinking": False} if is_qwen else {}

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        **template_kwargs,
    )

    if hasattr(inputs, "keys"):
        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }
    else:
        inputs = {"input_ids": inputs.to(model.device)}

    prompt_length = inputs["input_ids"].shape[-1]

    eos_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<turn|>"),
    ]
    eos_ids = list(dict.fromkeys(
        token for token in eos_ids
        if isinstance(token, int) and token >= 0
    ))

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated_ids = outputs[0, prompt_length:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()
