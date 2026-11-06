import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint

from src.data.preprocessing import prepare_translation_examples
from src.data.tokenization import tokenize_translation
from src.data.collator import MTDataCollator
from src.models.load_model import load_model_and_tokenizer
from src.models.lora import apply_lora
from src.utils.training_monitor import TrainingMonitor
from src.training.mt_metrics import compute_mt_metrics


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
        if logs:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step": state.global_step,
                    **logs,
                }) + "\n")



class MTGenerationCallback(TrainerCallback):
    def __init__(self, model, tokenizer, examples, max_length=512, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.examples = examples
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.no_grad()
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        from time import perf_counter
        from tqdm.auto import tqdm

        if metrics is None:
            raise RuntimeError("Evaluation metrics are missing")

        model = self.model
        tokenizer = self.tokenizer
        was_training = model.training
        old_padding_side = tokenizer.padding_side
        old_use_cache = model.config.use_cache

        predictions = []
        references = []

        is_qwen = "qwen" in tokenizer.name_or_path.lower()
        template_kwargs = {"enable_thinking": False} if is_qwen else {}

        tokenizer.padding_side = "left"
        model.config.use_cache = True
        model.eval()
        started = perf_counter()

        try:
            for start in tqdm(
                range(0, len(self.examples), self.batch_size),
                desc="Generating translations",
            ):
                batch = self.examples[start:start + self.batch_size]

                prompts = [
                    tokenizer.apply_chat_template(
                        [{
                            "role": "user",
                            "content": (
                                f'{example["instruction"]}\\n\\n'
                                f'{example["input"]}'
                            ),
                        }],
                        tokenize=False,
                        add_generation_prompt=True,
                        **template_kwargs,
                    )
                    for example in batch
                ]

                encoded = tokenizer(
                    prompts,
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(model.device)

                if encoded["input_ids"].shape[1] > self.max_length:
                    raise ValueError("Validation prompt too long")

                generated = model.generate(
                    **encoded,
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

                prompt_length = encoded["input_ids"].shape[1]
                new_tokens = generated[:, prompt_length:]

                predictions.extend(
                    text.strip()
                    for text in tokenizer.batch_decode(
                        new_tokens,
                        skip_special_tokens=True,
                    )
                )

                references.extend(
                    example["output"].strip()
                    for example in batch
                )

        finally:
            tokenizer.padding_side = old_padding_side
            model.config.use_cache = old_use_cache
            if was_training:
                model.train()

        scores = compute_mt_metrics(predictions, references)
        duration = perf_counter() - started

        metrics["eval_chrf"] = scores["chrf"]
        metrics["eval_bleu"] = scores["bleu"]

        with open(
            Path(args.output_dir) / "mt_generation_metrics.jsonl",
            "a",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps({
                "step": state.global_step,
                "chrf": scores["chrf"],
                "bleu": scores["bleu"],
                "generation_seconds": round(duration, 2),
                "batch_size": self.batch_size,
                "samples": len(predictions),
            }) + "\\n")

        print(
            f"Step {state.global_step}: "
            f"chrF++={scores['chrf']:.4f}, "
            f"BLEU={scores['bleu']:.4f}, "
            f"generation={duration:.1f}s",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/models/qwen3_5_4b_lora.yaml",
    )
    parser.add_argument(
        "--language",
        choices=["gusii", "kamba", "luhya", "somali"],
        default="kamba",
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
    lora_config = config["lora"]
    training_config = config["training"]

    max_length = (
        args.max_length
        if args.max_length is not None
        else training_config["max_length"]
    )

    print("Model:", model_name)
    print("Language:", args.language)
    print("Max length:", max_length)
    print("LoRA:", lora_config)
    print("Training:", training_config)

    if args.check_config:
        print("Configuration: OK")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    if model_name not in [
        "Qwen/Qwen3.5-4B",
        "McGill-NLP/AfriqueQwen3.5-4B",
        "google/gemma-4-E4B-it",
    ]:
        raise RuntimeError(
            "LoRA architecture not yet validated for this model"
        )

    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max_steps must be positive")

    random.seed(training_config["seed"])
    torch.manual_seed(training_config["seed"])

    examples = prepare_translation_examples(
        "data_construction/processed/multilingual/train.jsonl",
        args.language,
    )

    random.Random(training_config["seed"]).shuffle(examples)

    if args.max_samples > 0:
        examples = examples[:args.max_samples]

    if not examples:
        raise RuntimeError("No training examples selected")

    print("Selected examples:", len(examples))

    model, tokenizer = load_model_and_tokenizer(
        model_name,
        local_files_only=config.get("local_files_only", False),
        torch_dtype=config.get("torch_dtype", "bfloat16"),
    )

    model = apply_lora(
        model,
        rank=lora_config["rank"],
        alpha=lora_config["alpha"],
        dropout=lora_config["dropout"],
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    tokenized = []
    skipped = 0

    for example in examples:
        try:
            tokenized.append(
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

    print("Tokenized examples:", len(tokenized))
    print("Skipped long examples:", skipped)

    if not tokenized:
        raise RuntimeError("No valid training examples")


    validation_examples = prepare_translation_examples(
        "data_construction/processed/multilingual/validation.jsonl",
        args.language,
    )

    if len(validation_examples) % 2:
        raise RuntimeError("Unpaired validation examples")

    original_pairs = [
        validation_examples[i:i + 2]
        for i in range(0, len(validation_examples), 2)
    ]

    rng = random.Random(training_config["seed"])
    rng.shuffle(original_pairs)

    if len(original_pairs) < 100:
        raise RuntimeError("Fewer than 100 original validation sentences")

    selected_pairs = original_pairs[:100]

    validation_generation_examples = []

    validation_tokenized = []
    skipped_validation = 0

    for pair in selected_pairs:
        encoded_pair = []

        for example in pair:
            try:
                encoded_pair.append(
                    tokenize_translation(
                        example,
                        tokenizer,
                        max_length=max_length,
                    )
                )
            except ValueError as error:
                if "Sequence too long" not in str(error):
                    raise
                skipped_validation += 1

        # Conserver uniquement les phrases avec les deux directions
        if len(encoded_pair) == 2:
            validation_tokenized.extend(encoded_pair)
            validation_generation_examples.extend(pair)

    if not validation_tokenized:
        raise RuntimeError("No valid validation examples")

    print("Validation original sentences:", len(validation_tokenized) // 2)
    print("Validation translation examples:", len(validation_tokenized))
    print("Skipped validation directions:", skipped_validation)

    model_slug = model_name.split("/")[-1].lower()
    run_name = (
        f"{model_slug}-lora-"
        f"{args.max_steps if args.max_steps is not None else 'epochs'}-"
        f"{len(tokenized)}"
    )

    model_dirs = {
        "Qwen/Qwen3.5-4B": "qwen3_5_4b",
        "McGill-NLP/AfriqueQwen3.5-4B": "afriqueqwen3_5_4b",
        "google/gemma-4-E4B-it": "gemma4_e4b",
    }

    output_dir = (
        Path("/network/scratch/a/ahomagnr/MT_Project/outputs")
        / model_dirs[model_name]
        / args.language
        / run_name
    )

    if args.output_dir:
        output_dir = Path(args.output_dir) / run_name

    resume_checkpoint = None

    if output_dir.exists():
        resume_checkpoint = get_last_checkpoint(str(output_dir))

        if resume_checkpoint is None:
            if any(output_dir.iterdir()):
                raise RuntimeError(
                    f"Existing output directory without checkpoint: {output_dir}"
                )
        else:
            print("Resuming from checkpoint:", resume_checkpoint)

    training_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=training_config[
            "per_device_train_batch_size"
        ],
        gradient_accumulation_steps=training_config[
            "gradient_accumulation_steps"
        ],
        learning_rate=training_config["learning_rate"],
        num_train_epochs=training_config["num_train_epochs"],
        warmup_steps=training_config["warmup_steps"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        bf16=training_config["bf16"],
        gradient_checkpointing=training_config[
            "gradient_checkpointing"
        ],
        logging_strategy="steps",
        logging_steps=config["evaluation"]["logging_steps"],
        eval_strategy=config["evaluation"]["strategy"],
        eval_steps=config["evaluation"]["eval_steps"],
        per_device_eval_batch_size=training_config[
            "per_device_eval_batch_size"
        ],
        save_strategy=config["checkpoint"]["save_strategy"],
        save_steps=config["checkpoint"]["save_steps"],
        save_total_limit=config["checkpoint"]["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=config["checkpoint"]["report_to"],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=training_config["seed"],
    )

    if args.max_steps is not None:
        training_kwargs["max_steps"] = args.max_steps

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TranslationDataset(tokenized),
        eval_dataset=TranslationDataset(validation_tokenized),
        data_collator=MTDataCollator(tokenizer),
        processing_class=tokenizer,
        callbacks=[
            MetricsCallback(output_dir),
            EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.0,
            ),
        ],
    )

    monitor = TrainingMonitor()
    monitor.start()

    try:
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    finally:
        monitor.stop(
            output_dir=output_dir,
            language=args.language,
            steps=int(trainer.state.global_step),
        )

    # Evaluation finale uniquement, après restauration du meilleur checkpoint
    MTGenerationCallback(
        model=trainer.model,
        tokenizer=tokenizer,
        examples=validation_generation_examples,
        max_length=max_length,
    ).on_evaluate(
        args=training_args,
        state=trainer.state,
        control=trainer.control,
        metrics={},
    )

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print("Training completed:", output_dir)
    print("Steps:", train_result.global_step)


if __name__ == "__main__":
    main()
