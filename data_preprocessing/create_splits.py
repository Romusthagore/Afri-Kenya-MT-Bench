
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "native_file" / "dataset.xlsx"

train_sheets = [
    "Chat Train", "Chat Valid", "Chat Test",
    "Health Train", "Health Valid", "Health Test",
    "News Train", "News Valid", "News Test",
    "nllb_seed"
]

train = pd.concat(
    [pd.read_excel(FILE, sheet_name=s) for s in train_sheets],
    ignore_index=True
)

val = pd.read_excel(FILE, sheet_name="flores200_dev")
test = pd.read_excel(FILE, sheet_name="flores200_devtest")

train.to_csv(ROOT / "train.csv", index=False)
val.to_csv(ROOT / "validation.csv", index=False)
test.to_csv(ROOT / "test.csv", index=False)

print("Train:", train.shape)
print("Validation:", val.shape)
print("Test:", test.shape)


