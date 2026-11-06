import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3_5ForConditionalGeneration,
    Gemma4ForConditionalGeneration,
)


def load_model_and_tokenizer(
    model_name,
    local_files_only=False,
    torch_dtype="bfloat16",
):
    dtype = getattr(torch, torch_dtype)

    config = AutoConfig.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        local_files_only=local_files_only,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_classes = {
        "qwen3_5": Qwen3_5ForConditionalGeneration,
        "gemma4": Gemma4ForConditionalGeneration,
    }

    model_class = model_classes.get(
        config.model_type,
        AutoModelForCausalLM,
    )

    print("Architecture:", config.model_type)
    print("Model class:", model_class.__name__)

    model = model_class.from_pretrained(
        model_name,
        dtype=dtype,
        local_files_only=local_files_only,
    )

    return model, tokenizer
