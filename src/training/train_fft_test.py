import argparse
import json
import random
from pathlib import Path

import torch
import yaml
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint

from src.data.preprocessing import prepare_translation_examples
from src.data.tokenization import tokenize_translation
from src.data.collator import MTDataCollator


MODELS = {
    "Qwen/Qwen3.5-4B": (
        "qwen3_5_4b",
        "Qwen3_5ForConditionalGeneration",
    ),
    "McGill-NLP/AfriqueQwen3.5-4B": (
        "afriqueqwen3_5_4b",
        "Qwen3_5ForConditionalGeneration",
    ),
    "google/gemma-4-E4B-it": (
        "gemma4_e4b",
        "Gemma4ForConditionalGeneration",
    ),
}


class TranslationDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class MetricsCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.path = Path(output_dir) / "training_history.jsonl"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "step": state.global_step,
                **logs,
            }) + "\n")


def load_model(model_name, local_files_only=False):
    _, class_name = MODELS[model_name]

    model_class = getattr(transformers, class_name, None)

    if model_class is None:
        raise RuntimeError(
            f"{class_name} unavailable in Transformers "
            f"{transformers.__version__}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        local_files_only=local_files_only,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Accelerate FSDP2 gère le chargement distribué.
    model = model_class.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False

    # Full fine-tuning du chemin textuel.
    # L'encodeur vision reste gelé.
    for name, parameter in model.named_parameters():
        is_vision = (
            "visual" in name.lower()
            or "vision_tower" in name.lower()
            or "vision_model" in name.lower()
        )
        parameter.requires_grad_(not is_vision)

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in model.parameters())

    print(
        f"Trainable parameters: {trainable:,}/{total:,}",
        flush=True,
    )

    return model, tokenizer


def prepare_data(path, language, tokenizer, max_length):
    examples = prepare_translation_examples(path, language)

    tokenized = []
    retained = []
    skipped = 0

    if len(examples) % 2:
        raise RuntimeError("Unpaired translation examples")

    for i in range(0, len(examples), 2):
        pair = examples[i:i + 2]
        encoded = []

        for example in pair:
            try:
                encoded.append(
                    tokenize_translation(
                        example,
                        tokenizer,
                        max_length=max_length,
                    )
                )
            except ValueError as error:
                if "Sequence too long" not in str(error):
                    raise
                skipped += 1

        # Conserver uniquement les paires complètes.
        if len(encoded) == 2:
            tokenized.extend(encoded)
            retained.extend(pair)

    return tokenized, retained, skipped


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="NEW/configs/models/qwen3_5_4b_lora.yaml",
    )
    parser.add_argument(
        "--language",
        choices=["gusii", "kamba", "luhya", "somali"],
        default="luhya",
    )
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--check_config", action="store_true")
    parser.add_argument("--output_dir", default=None)

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]

    if model_name not in MODELS:
        raise ValueError(f"Unsupported model: {model_name}")

    training = config["training"]
    evaluation = config["evaluation"]
    checkpoint = config["checkpoint"]

    seed = training["seed"]
    max_length = args.max_length or training["max_length"]

    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max_steps must be positive")

    if args.max_samples < 0:
        raise ValueError("--max_samples must be nonnegative")

    print("Model:", model_name)
    print("Language:", args.language)
    print("Method: Full fine-tuning")
    print("Max length:", max_length)
    print("Transformers:", transformers.__version__)

    if args.check_config:
        print("Configuration: OK")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    random.seed(seed)
    torch.manual_seed(seed)

    model, tokenizer = load_model(
        model_name,
        local_files_only=config.get("local_files_only", False),
    )

    train_data, _, train_skipped = prepare_data(
        str(Path(__file__).resolve().parents[3] / "data_construction/processed/multilingual/train.jsonl"),
        args.language,
        tokenizer,
        max_length,
    )

    rng = random.Random(seed)

    # Shuffle by sentence pair, preserving both directions.
    train_pairs = [
        train_data[i:i + 2]
        for i in range(0, len(train_data), 2)
    ]
    rng.shuffle(train_pairs)

    if args.max_samples:
        train_pairs = train_pairs[:args.max_samples // 2]

    train_data = [
        item for pair in train_pairs for item in pair
    ]

    val_data, val_examples, val_skipped = prepare_data(
        str(Path(__file__).resolve().parents[3] / "data_construction/processed/multilingual/validation.jsonl"),
        args.language,
        tokenizer,
        max_length,
    )

    if not train_data or not val_data:
        raise RuntimeError("Empty training or validation dataset")

    print("Training examples:", len(train_data))
    print("Validation examples:", len(val_data))
    print("Skipped train directions:", train_skipped)
    print("Skipped validation directions:", val_skipped)

    model_dir = MODELS[model_name][0]
    run_name = (
        f"{model_dir}-fft-"
        f"{args.max_steps or 'epochs'}-"
        f"{len(train_data)}"
    )

    output_root = Path(
        args.output_dir or str(Path(__file__).resolve().parents[2] / "outputs")
    )

    output_dir = (
        output_root / model_dir / args.language / run_name
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    resume = get_last_checkpoint(str(output_dir))

    if resume is None and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Nonempty output directory without checkpoint: {output_dir}"
        )

    training_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=training[
            "per_device_train_batch_size"
        ],
        per_device_eval_batch_size=training[
            "per_device_eval_batch_size"
        ],
        gradient_accumulation_steps=training[
            "gradient_accumulation_steps"
        ],
        learning_rate=training["learning_rate"],
        num_train_epochs=training["num_train_epochs"],
        warmup_steps=training["warmup_steps"],
        lr_scheduler_type=training["lr_scheduler_type"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
        logging_strategy="steps",
        logging_steps=evaluation["logging_steps"],
        eval_strategy=evaluation["strategy"],
        eval_steps=evaluation["eval_steps"],
        save_strategy="no",
        save_steps=checkpoint["save_steps"],
        save_total_limit=checkpoint["save_total_limit"],
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=seed,
        save_safetensors=True,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    if args.max_steps is not None:
        training_kwargs["max_steps"] = args.max_steps

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TranslationDataset(train_data),
        eval_dataset=TranslationDataset(val_data),
        data_collator=MTDataCollator(tokenizer),
        processing_class=tokenizer,
        callbacks=[MetricsCallback(output_dir)],
    )

    result = trainer.train(resume_from_checkpoint=resume)

    # Save full model using Trainer's distributed-safe mechanism.
    # No final save during test

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir / "final_model")

        with (output_dir / "run_summary.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump({
                "model": model_name,
                "language": args.language,
                "method": "full_fine_tuning",
                "steps": result.global_step,
                "train_examples": len(train_data),
                "validation_examples": len(val_data),
                "learning_rate": 1e-5,
            }, f, indent=2)

        print("Training completed:", output_dir)
        print("Steps:", result.global_step)


if __name__ == "__main__":
    main()
