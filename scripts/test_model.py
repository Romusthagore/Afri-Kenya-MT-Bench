import torch

from src.models.load_model import load_model_and_tokenizer
from src.models.lora import apply_lora

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

print("GPU:", torch.cuda.get_device_name(0))

model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

print("Model loaded:", model.config.model_type)
print("Tokenizer loaded:", len(tokenizer))

model = apply_lora(model)
model.to("cuda")

print("LoRA applied successfully")
print("Model device:", next(model.parameters()).device)
