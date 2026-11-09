def tokenize_translation(example, tokenizer, max_length=512):
    user_text = (
        f'{example["instruction"]}\n\n'
        f'{example["input"]}'
    )

    prompt = [{"role": "user", "content": user_text}]
    conversation = prompt + [
        {"role": "assistant", "content": example["output"]}
    ]

    is_qwen = "qwen" in tokenizer.name_or_path.lower()
    template_kwargs = {"enable_thinking": False} if is_qwen else {}

    prompt_ids = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        **template_kwargs,
    )["input_ids"]

    full_ids = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=False,
        **template_kwargs,
    )["input_ids"]

    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt tokens do not match conversation prefix")

    if len(full_ids) > max_length:
        raise ValueError(
            f"Sequence too long: {len(full_ids)} > {max_length}"
        )

    labels = [-100] * len(prompt_ids)
    labels += full_ids[len(prompt_ids):]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }
