from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_TOKEN = os.getenv("HF_TOKEN")

t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    dtype=torch.float16,
    device_map={"": "mps"},
    low_cpu_mem_usage=True
)

load_time = time.time() - t0

print(f"Model loaded in {load_time:.2f} seconds")
print("Model device:", next(model.parameters()).device)
