import torch


class MTDataCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_length = max(len(x["input_ids"]) for x in features)

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for example in features:
            padding = max_length - len(example["input_ids"])

            batch["input_ids"].append(
                example["input_ids"] + [self.pad_token_id] * padding
            )

            batch["attention_mask"].append(
                example["attention_mask"] + [0] * padding
            )

            batch["labels"].append(
                example["labels"] + [-100] * padding
            )

        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in batch.items()
        }
