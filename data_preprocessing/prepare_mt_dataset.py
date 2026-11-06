
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "processed"

LANGUAGES = {
    "gusii": ["Gusii", "Gusii_A", "Gusii_B"],
    "kamba": ["Kamba", "Kamba_A", "Kamba_B"],
    "luhya": ["Luhya", "Luhya_A", "Luhya_B"],
    "somali": ["Somali", "Somali_A", "Somali_B"],
}

SPLITS = {
    "train": ROOT / "train.csv",
    "validation": ROOT / "validation.csv",
    "test": ROOT / "test.csv",
}


def clean(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value if value else None


def get_references(row, columns):
    references = []
    seen = set()

    for col in columns:
        if col not in row.index:
            continue

        value = clean(row[col])
        if value is None or value in seen:
            continue

        references.append(value)
        seen.add(value)

    return references


def convert_split(df):
    records = {lang: [] for lang in LANGUAGES}

    for row_number, row in df.iterrows():
        english = clean(row.get("English text"))

        if english is None:
            continue

        for lang, columns in LANGUAGES.items():
            references = get_references(row, columns)

            if not references:
                continue

            record = {
                "id": f"{row_number + 1}",
                "domain": clean(row.get("Domain")),
                "english_id": clean(row.get("English ID")),
                "source_lang": "english",
                "target_lang": lang,
                "source_text": english,
                "target_text": references[0],
                "references": references,
            }

            records[lang].append(record)

    return records


def save_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    for split, csv_path in SPLITS.items():
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        by_language = convert_split(df)

        multilingual = []

        print(f"\n{split.upper()}")

        for lang, records in by_language.items():
            save_jsonl(
                OUTPUT / "by_language" / lang / f"{split}.jsonl",
                records,
            )

            multilingual.extend(records)

            print(f"  {lang}: {len(records)} examples")

        save_jsonl(
            OUTPUT / "multilingual" / f"{split}.jsonl",
            multilingual,
        )

        print(f"  multilingual: {len(multilingual)} examples")


if __name__ == "__main__":
    main()

