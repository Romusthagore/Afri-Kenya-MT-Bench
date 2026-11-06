#!/usr/bin/env python
"""Évaluation MT par direction, avec sauvegarde des prédictions."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def stop_token_ids(tokenizer):
    vocab = tokenizer.get_vocab()
    ids = []

    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)

    for token in ("<|im_end|>", "<turn|>", "<end_of_turn>"):
        if token in vocab:
            ids.append(vocab[token])

    return list(dict.fromkeys(ids))


def build_prompt(tokenizer, example):
    is_qwen = "qwen" in tokenizer.name_or_path.lower()
    kwargs = {"enable_thinking": False} if is_qwen else {}

    return tokenizer.apply_chat_template(
        [{
            "role": "user",
            "content": f'{example["instruction"]}\n\n{example["input"]}',
        }],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def generate_batch(model, tokenizer, prompts, max_new_tokens, stop_ids):
    import torch

    encoded = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    generated = outputs[:, encoded["input_ids"].shape[1]:]

    predictions = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )

    # Ne pas confondre le padding avec un véritable token de fin.
    truncated = []
    for row in generated:
        ids = row.tolist()
        if stop_ids:
            ended = any(token in stop_ids for token in ids)
            truncated.append(not ended and len(ids) >= max_new_tokens)
        else:
            truncated.append(len(ids) >= max_new_tokens)

    return [text.strip() for text in predictions], truncated


def summarize(rows):
    from src.evaluation.metrics import compute_translation_metrics

    predictions = [r["prediction"] for r in rows]
    references = [r["reference"] for r in rows]
    sources = [r["source"] for r in rows]

    scores = compute_translation_metrics(predictions, references)
    n = len(rows)

    scores.update({
        "truncated_pct": round(
            100 * sum(r["truncated"] for r in rows) / n, 2
        ),
        "empty_pct": round(
            100 * sum(not p for p in predictions) / n, 2
        ),
        "copy_source_pct": round(
            100 * sum(p == s for p, s in zip(predictions, sources)) / n, 2
        ),
        "pred_over_ref_length": round(
            sum(len(p) for p in predictions)
            / max(1, sum(len(r) for r in references)), 3
        ),
    })

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--language", default="luhya")
    parser.add_argument(
        "--dataset",
        default="data_construction/processed/multilingual/validation.jsonl",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Nombre de paires par direction ; 0 = toutes")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()

    from src.data.preprocessing import prepare_translation_examples
    from src.data.tokenization import tokenize_translation
    from src.evaluation.generate import load_translation_model

    examples = prepare_translation_examples(args.dataset, args.language)

    model, tokenizer = load_translation_model(args.config, args.adapter)
    tokenizer.padding_side = "left"

    # Filtrage par paires : même phrase dans les deux directions.
    pairs = []

    for i in range(0, len(examples), 2):
        pair = examples[i:i + 2]

        if len(pair) != 2:
            continue

        try:
            for example in pair:
                tokenize_translation(
                    example, tokenizer, max_length=args.max_length
                )
        except ValueError as error:
            if "Sequence too long" not in str(error):
                raise
            continue

        pairs.append(pair)

    rng = random.Random(args.seed)

    if args.max_samples and len(pairs) > args.max_samples:
        indices = sorted(rng.sample(range(len(pairs)), args.max_samples))
        pairs = [pairs[i] for i in indices]

    selected = [example for pair in pairs for example in pair]

    if not selected:
        raise RuntimeError("Aucun exemple disponible pour l'évaluation")

    print(f"Language: {args.language}")
    print(f"Pairs: {len(pairs)}")
    print(f"Examples: {len(selected)}")

    stop_ids = stop_token_ids(tokenizer)
    print("Stop token IDs:", stop_ids)

    prompts = [build_prompt(tokenizer, example) for example in selected]

    # Trier par longueur de tokens, pas par nombre de caractères.
    lengths = [
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for prompt in prompts
    ]
    order = sorted(range(len(selected)), key=lambda i: lengths[i], reverse=True)

    results = {}
    start_time = time.perf_counter()

    for start in range(0, len(order), args.batch_size):
        indices = order[start:start + args.batch_size]

        predictions, truncated = generate_batch(
            model,
            tokenizer,
            [prompts[i] for i in indices],
            args.max_new_tokens,
            stop_ids,
        )

        for i, prediction, cut in zip(indices, predictions, truncated):
            example = selected[i]

            results[i] = {
                "id": example["id"],
                "direction": (
                    f'{example["source_lang"]}_to_{example["target_lang"]}'
                ),
                "source": example["input"],
                "reference": example["output"],
                "prediction": prediction,
                "truncated": cut,
            }

        done = min(start + args.batch_size, len(order))
        print(f"{done}/{len(order)}", flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    ordered_results = [results[i] for i in range(len(selected))]

    with (output / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in ordered_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    directions = [
        f"english_to_{args.language}",
        f"{args.language}_to_english",
    ]

    per_direction = {}

    for direction in directions:
        subset = [
            row for row in ordered_results
            if row["direction"] == direction
        ]

        if subset:
            per_direction[direction] = summarize(subset)

    report = {
        "config": args.config,
        "adapter": args.adapter,
        "dataset": args.dataset,
        "language": args.language,
        "num_pairs": len(pairs),
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "generation_seconds": round(
            time.perf_counter() - start_time, 2
        ),
        "per_direction": per_direction,
        "mixed_all_directions": summarize(ordered_results),
    }

    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nRESULTS")
    for direction, scores in per_direction.items():
        print(
            f'{direction}: '
            f'BLEU={scores["bleu"]:.2f}, '
            f'chrF++={scores["chrf++"]:.2f}, '
            f'n={scores["num_examples"]}'
        )

    if args.show:
        for direction in directions:
            subset = [
                row for row in ordered_results
                if row["direction"] == direction
            ]
            print(f"\n{direction}")
            for row in subset[:args.show]:
                print("SOURCE:", row["source"])
                print("REFERENCE:", row["reference"])
                print("PREDICTION:", row["prediction"])
                print()

    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
