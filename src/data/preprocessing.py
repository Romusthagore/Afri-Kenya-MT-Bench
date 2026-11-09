from .dataset import load_mt_dataset


def prepare_translation_examples(path, language):
    """Prépare les exemples SFT pour une langue, dans les deux directions."""
    dataset = load_mt_dataset(path)
    examples = []

    for row in dataset:
        if row["target_lang"].lower() != language.lower():
            continue

        english = row["source_text"]
        translation = row["target_text"]

        # English -> langue africaine
        examples.append({
            "id": f'{row["id"]}_en_to_{language}',
            "source_lang": "english",
            "target_lang": language,
            "instruction": f"Translate the following text from English to {language.capitalize()}. Return only the translation.",
            "input": english,
            "output": translation,
        })

        # Langue africaine -> English
        examples.append({
            "id": f'{row["id"]}_{language}_to_en',
            "source_lang": language,
            "target_lang": "english",
            "instruction": f"Translate the following text from {language.capitalize()} to English. Return only the translation.",
            "input": translation,
            "output": english,
        })

    return examples


if __name__ == "__main__":
    path = "data_construction/processed/multilingual/train.jsonl"
    language = "kamba"

    examples = prepare_translation_examples(path, language)

    print("Language:", language)
    print("Total examples:", len(examples))
    print("First example:", examples[0])
    print("Second example:", examples[1])
