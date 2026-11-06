import json
from pathlib import Path

import torch
import torch.distributed as dist

from src.training.mt_metrics import compute_mt_metrics


@torch.inference_mode()
def evaluate_translation(
    trainer,
    tokenizer,
    examples,
    output_dir,
    batch_size=1,
    max_new_tokens=128,
):
    model = trainer.model_wrapped
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    was_training = model.training
    old_padding_side = tokenizer.padding_side
    old_use_cache = model.config.use_cache

    tokenizer.padding_side = "left"
    model.config.use_cache = False
    model.eval()

    is_qwen = "qwen" in tokenizer.name_or_path.lower()
    template_kwargs = {"enable_thinking": False} if is_qwen else {}

    local_results = []

    try:
        # Tous les processus exécutent le même nombre de générations.
        total_batches = (
            len(examples) + batch_size * world_size - 1
        ) // (batch_size * world_size)

        for batch_index in range(total_batches):
            start = (
                batch_index * world_size + rank
            ) * batch_size

            batch = examples[start:start + batch_size]

            if not batch:
                # Maintenir les appels collectifs FSDP synchronisés.
                batch = [examples[-1]]
                indices = [-1]
            else:
                indices = list(
                    range(start, start + len(batch))
                )

            prompts = [
                tokenizer.apply_chat_template(
                    [{
                        "role": "user",
                        "content": (
                            f'{example["instruction"]}\n\n'
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
            ).to(trainer.args.device)

            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )

            new_tokens = generated[:, encoded["input_ids"].shape[1]:]

            predictions = tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True,
            )

            for index, prediction in zip(indices, predictions):
                if index >= 0:
                    local_results.append((index, prediction.strip()))

        if dist.is_initialized():
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_results)
            results = [
                item
                for group in gathered
                for item in group
            ]
        else:
            results = local_results

        if rank == 0:
            results.sort(key=lambda x: x[0])

            if len(results) != len(examples):
                raise RuntimeError("Incomplete validation predictions")

            predictions = [prediction for _, prediction in results]
            references = [example["output"].strip() for example in examples]

            languages = {
                ex["target_lang"] if ex["source_lang"] == "english"
                else ex["source_lang"]
                for ex in examples
            }
            if len(languages) != 1:
                raise ValueError(f"Expected one language, got: {languages}")
            language = languages.pop()

            groups = {
                "overall": list(range(len(examples))),
                f"en_to_{language}": [
                    i for i, ex in enumerate(examples)
                    if ex["source_lang"] == "english"
                ],
                f"{language}_to_en": [
                    i for i, ex in enumerate(examples)
                    if ex["target_lang"] == "english"
                ],
            }

            metrics = {}

            for name, indices in groups.items():
                scores = compute_mt_metrics(
                    [predictions[i] for i in indices],
                    [references[i] for i in indices],
                )

                metrics[name] = {
                    "bleu": scores["bleu"],
                    "chrf": scores["chrf"],
                    "samples": len(indices),
                }

            path = Path(output_dir) / "validation_metrics.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(metrics, indent=2),
                encoding="utf-8",
            )

            print(json.dumps(metrics, indent=2), flush=True)
        else:
            metrics = None

        return metrics

    finally:
        tokenizer.padding_side = old_padding_side
        model.config.use_cache = old_use_cache

        if was_training:
            model.train()
