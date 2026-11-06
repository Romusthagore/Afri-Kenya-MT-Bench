import torch
from peft import LoraConfig, TaskType, get_peft_model


def apply_lora(model, rank=8, alpha=16, dropout=0.05):
    target_modules = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith("model.language_model.layers.")
    ]

    if not target_modules:
        raise ValueError("No language model linear layers found")

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(
        model,
        config,
        autocast_adapter_dtype=True,
    )

    model.print_trainable_parameters()
    return model
