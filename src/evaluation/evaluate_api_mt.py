"""
Afri-Kenya-MT-Bench: API translation evaluation.

Commands:
    run   - Generate translations
    score - Compute BLEU and chrF++
    sweep - Run multiple configurations

Models:
    gpt, claude

Languages:
    gusii, kamba, luhya, somali

Directions:
    forward, reverse

Shot settings:
    0, 5
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from src.evaluation.api_clients import AzureMTClient
from src.evaluation.mt_sft import (
    load_jsonl,
    select_examples,
    build_prompt,
)
from src.evaluation.metrics import compute_translation_metrics


LANGUAGES = ["gusii", "kamba", "luhya", "somali"]
MODELS = ["gpt", "claude"]
DIRECTIONS = ["forward", "reverse"]
SHOTS = [0, 5]

DEFAULT_DATA_ROOT = Path(
    "/network/scratch/a/ahomagnr/MT_Project/"
    "data_construction/processed/multilingual"
)

DEFAULT_RESULTS_ROOT = Path("results/api_mt_sft")


def experiment_name(model, language, direction, shots):
    return f"{model}_{language}_{direction}_{shots}shot"


def experiment_path(args):
    name = experiment_name(
        args.model,
        args.language,
        args.direction,
        args.shots,
    )
    return Path(args.results_root) / "predictions" / f"{name}.jsonl"


def read_predictions(path):
    if not path.exists():
        return []

    return load_jsonl(path)


def get_completed_ids(path):
    completed = set()

    for row in read_predictions(path):
        if row.get("status") == "success":
            completed.add(str(row["id"]))
        elif (
            "content_filter" in row.get("error", "").lower()
            or "content policy" in row.get("error", "").lower()
        ):
            completed.add(str(row["id"]))

    return completed


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )
        file.flush()


def run_experiment(args):
    train_path = Path(args.data_root) / "validation.jsonl"
    test_path = Path(args.data_root) / "test.jsonl"

    test_rows = select_examples(
        test_path, args.language, args.direction
    )

    train_rows = (
        select_examples(train_path, args.language, args.direction)
        if args.shots == 5 else None
    )

    if args.limit is not None:
        test_rows = test_rows[:args.limit]

    output_path = experiment_path(args)
    completed = get_completed_ids(output_path)

    print("=" * 65)
    print("AFRI-KENYA MT BENCHMARK")
    print("=" * 65)
    print("Model:", args.model)
    print("Language:", args.language)
    print("Direction:", args.direction)
    print("Shots:", args.shots)
    print("Test examples:", len(test_rows))
    print("Already completed:", len(completed))
    print("Output:", output_path)
    print("=" * 65, flush=True)

    client = AzureMTClient(
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
    )

    successes = 0
    failures = 0
    skipped = 0

    try:
        for index, row in enumerate(test_rows, start=1):
            row_id = str(row.get("id", index))

            if row_id in completed:
                skipped += 1
                continue

            record = {
                "id": row_id,
                "language": args.language,
                "direction": args.direction,
                "shots": args.shots,
                "model": args.model,
                "source_text": row["input"],
                "references": [row["output"]],
                "prediction": None,
                "status": "failed",
            }

            try:
                prompt = build_prompt(
                    example=row,
                    train_examples=train_rows,
                    shots=args.shots,
                    seed=args.seed,
                )

                response = client.translate(prompt)

                record.update(asdict(response))
                record["prediction"] = response.text
                record["status"] = "success"

                successes += 1
                completed.add(row_id)

                print(
                    f"[{index}/{len(test_rows)}] OK "
                    f"id={row_id} "
                    f"latency={response.latency_seconds:.2f}s",
                    flush=True,
                )

            except Exception as exc:
                failures += 1
                record["error_type"] = type(exc).__name__
                record["error"] = str(exc)[:500]

                print(
                    f"[{index}/{len(test_rows)}] ERROR "
                    f"id={row_id}: {type(exc).__name__}",
                    flush=True,
                )

            append_jsonl(output_path, record)

            if args.delay > 0:
                time.sleep(args.delay)

    finally:
        client.close()

    print("\nFinished")
    print("Successes:", successes)
    print("Failures:", failures)
    print("Skipped:", skipped)
    print("Predictions:", output_path)


def score_experiment(args):
    path = experiment_path(args)
    rows = read_predictions(path)

    # Keep successful predictions, including after retries.
    latest = {}

    for row in rows:
        row_id = str(row["id"])

        if row_id not in latest:
            latest[row_id] = row

        if (
            row.get("status") == "success"
            and isinstance(row.get("prediction"), str)
            and row["prediction"].strip()
        ):
            latest[row_id] = row

    if not latest:
        raise ValueError(f"No predictions in {path}")

    predictions = []
    references = []
    successes = 0

    for row in latest.values():
        prediction = (
            row.get("prediction", "")
            if row.get("status") == "success"
            else ""
        )

        if not isinstance(prediction, str):
            prediction = ""

        if prediction.strip():
            successes += 1

        predictions.append(prediction)
        references.append(row["references"])

    metrics = compute_translation_metrics(
        predictions,
        references,
    )

    metrics["total_examples"] = len(latest)
    metrics["successful_examples"] = successes
    metrics["failed_examples"] = len(latest) - successes
    metrics["success_rate"] = round(
        100 * successes / len(latest), 2
    )

    metrics.update({
        "model": args.model,
        "language": args.language,
        "direction": args.direction,
        "shots": args.shots,
        "seed": args.seed,
        "prediction_file": str(path),
    })

    output_dir = Path(args.results_root) / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        experiment_name(
            args.model,
            args.language,
            args.direction,
            args.shots,
        ) + ".json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("Saved:", output_path)


def sweep_experiments(args):
    for model in MODELS:
        for language in LANGUAGES:
            for direction in DIRECTIONS:
                for shots in SHOTS:
                    args.model = model
                    args.language = language
                    args.direction = direction
                    args.shots = shots

                    run_experiment(args)

                    if args.score_after_run:
                        score_experiment(args)



def aggregate_experiments(args):
    """Compute macro-average BLEU and chrF++."""
    metrics_dir = Path(args.results_root) / "metrics"
    summary_dir = Path(args.results_root) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        for shots in SHOTS:
            rows = []
            missing = []

            for language in LANGUAGES:
                for direction in DIRECTIONS:
                    name = experiment_name(
                        model, language, direction, shots
                    )
                    path = metrics_dir / f"{name}.json"

                    if path.exists():
                        rows.append(json.loads(path.read_text(encoding="utf-8")))
                    else:
                        missing.append(name)

            if missing:
                print(f"{model} {shots}-shot: {len(missing)} missing experiments")
                for name in missing:
                    print("  Missing:", name)
                continue

            def average(group, key):
                return round(
                    sum(float(r[key]) for r in group) / len(group), 2
                )

            forward = [r for r in rows if r["direction"] == "forward"]
            reverse = [r for r in rows if r["direction"] == "reverse"]

            summary = {
                "model": model,
                "shots": shots,
                "aggregation": "macro_average",
                "overall": {
                    "bleu": average(rows, "bleu"),
                    "chrf": average(rows, "chrf"),
                    "samples": sum(r["total_examples"] for r in rows),
                },
                "forward": {
                    "bleu": average(forward, "bleu"),
                    "chrf": average(forward, "chrf"),
                    "samples": sum(r["total_examples"] for r in forward),
                },
                "reverse": {
                    "bleu": average(reverse, "bleu"),
                    "chrf": average(reverse, "chrf"),
                    "samples": sum(r["total_examples"] for r in reverse),
                },
            }

            output = summary_dir / f"{model}_{shots}shot_summary.json"
            output.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            print(json.dumps(summary, indent=2, ensure_ascii=False))
            print("Saved:", output)


def main():
    parser = argparse.ArgumentParser(
        description="Afri-Kenya MT API evaluation"
    )

    parser.add_argument(
        "command",
        choices=["run", "score", "sweep", "aggregate"],
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        default="gpt",
    )
    parser.add_argument(
        "--language",
        choices=LANGUAGES,
        default="kamba",
    )
    parser.add_argument(
        "--direction",
        choices=DIRECTIONS,
        default="forward",
    )
    parser.add_argument(
        "--shots",
        type=int,
        choices=SHOTS,
        default=0,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--score-after-run",
        action="store_true",
    )

    args = parser.parse_args()

    if args.command == "run":
        run_experiment(args)
    elif args.command == "score":
        score_experiment(args)
    elif args.command == "sweep":
        sweep_experiments(args)
    elif args.command == "aggregate":
        aggregate_experiments(args)


if __name__ == "__main__":
    main()
