
import random


def select_few_shots(train_examples, source_lang, target_lang, n_shots=5, seed=42):
    """Select reproducible in-context examples from the training set."""
    if n_shots < 0:
        raise ValueError("n_shots must be non-negative")

    candidates = [
        example
        for example in train_examples
        if example["source_lang"].lower() == source_lang.lower()
        and example["target_lang"].lower() == target_lang.lower()
    ]

    if len(candidates) < n_shots:
        raise ValueError(
            f"Not enough training examples for {source_lang} -> {target_lang}: "
            f"requested {n_shots}, available {len(candidates)}"
        )

    return random.Random(seed).sample(candidates, n_shots)


def build_translation_prompt(
    text,
    source_lang,
    target_lang,
    few_shots=None,
):
    """Build a zero-shot or few-shot machine translation prompt."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string")

    few_shots = few_shots or []

    source_name = source_lang.capitalize()
    target_name = target_lang.capitalize()

    instruction = (
        f"Translate the following text from {source_name} "
        f"to {target_name}. Return only the translation."
    )

    parts = [instruction]

    for example in few_shots:
        if (
            example["source_lang"].lower() != source_lang.lower()
            or example["target_lang"].lower() != target_lang.lower()
        ):
            raise ValueError("Few-shot example has an incorrect direction")

        parts.append(
            f"{source_name}: {example['input']}\n"
            f"{target_name}: {example['output']}"
        )

    parts.append(f"{source_name}: {text}\n{target_name}:")

    return "\n\n".join(parts)

