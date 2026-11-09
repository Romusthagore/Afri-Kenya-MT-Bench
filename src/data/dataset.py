import json
from pathlib import Path


def load_mt_dataset(path):
    path = Path(path)
    examples = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            example = json.loads(line)

            for field in ("source_text", "target_text", "target_lang"):
                if not example.get(field):
                    raise ValueError(
                        f"{path}:{line_number}: missing {field}"
                    )

            examples.append(example)

    return examples


if __name__ == "__main__":
    path = "data_construction/processed/multilingual/train.jsonl"
    dataset = load_mt_dataset(path)

    print("Total examples:", len(dataset))
    print("First example:", dataset[0])
