from fastapi import FastAPI
from pydantic import BaseModel
import time
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

# Load once at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    dtype=torch.float16,
    device_map={"": "mps"}, #metal performance shaders
    low_cpu_mem_usage=True
)
model.eval()

#prefix caching will not help much, but intra-request KV caching is doing all the work.

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    use_cache: bool = True  # Toggle KV-cache on/off

class GenerateResponse(BaseModel):
    text: str
    latency_ms: float
    tokens_per_sec: float
    timings: dict
    cache_enabled: bool

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    t_request_start = time.time()

    inputs = tokenizer(req.prompt, return_tensors="pt").to("mps")

    with torch.no_grad():
        if req.use_cache:
            # ---- WITH KV-CACHE ----
            t_prefill_start = time.time()

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=True
            )

            logits = outputs.logits
            past_key_values = outputs.past_key_values

            # First token
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            t_first_token = time.time()

            generated_ids = [next_token_id.item()]

            # Decode loop with cache
            for _ in range(req.max_new_tokens - 1):
                outputs = model(
                    input_ids=next_token_id,
                    past_key_values=past_key_values,
                    use_cache=True
                )

                logits = outputs.logits
                past_key_values = outputs.past_key_values

                next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids.append(next_token_id.item())

        else:
            # ---- WITHOUT KV-CACHE ----
            t_prefill_start = time.time()

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False
            )

            logits = outputs.logits
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
            t_first_token = time.time()

            generated_ids = [next_token_id.item()]

            # Decode loop WITHOUT cache (recomputes full sequence every time)
            for _ in range(req.max_new_tokens - 1):
                # Concatenate all previous tokens
                full_sequence = torch.cat([
                    inputs["input_ids"],
                    torch.tensor([generated_ids]).to("mps")
                ], dim=1)

                outputs = model(
                    input_ids=full_sequence,
                    use_cache=False
                )

                logits = outputs.logits
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids.append(next_token_id.item())

    t_done = time.time()

    prefill_ms = (t_first_token - t_prefill_start) * 1000
    ttft_ms = (t_first_token - t_request_start) * 1000
    decode_ms = (t_done - t_first_token) * 1000
    total_ms = (t_done - t_request_start) * 1000
    tokens_generated = len(generated_ids)

    avg_time_per_token_ms = decode_ms / max(tokens_generated - 1, 1)
    tokens_per_sec = tokens_generated / ((t_done - t_request_start)) if (t_done - t_request_start) > 0 else 0
    
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    cache_status = "WITH CACHE" if req.use_cache else "WITHOUT CACHE"
    print(f"\n{'='*50}")
    print(f"[MODE] {cache_status}")
    print(f"[Prefill] {prefill_ms:.2f} ms")
    print(f"[TTFT] {ttft_ms:.2f} ms (Time To First Token)")
    print(f"[Decode] {decode_ms:.2f} ms")
    print(f"[Tokens] {tokens_generated}")
    print(f"[Per-token] {avg_time_per_token_ms:.2f} ms/token")
    print(f"[Total] {total_ms:.2f} ms")
    print(f"[Throughput] {tokens_per_sec:.2f} tokens/sec")
    print(f"{'='*50}\n")

    return GenerateResponse(
        text=text,
        latency_ms=total_ms,
        tokens_per_sec=tokens_per_sec,
        cache_enabled=req.use_cache,
        timings={
            "prefill_ms": prefill_ms,
            "ttft_ms": ttft_ms,
            "decode_ms": decode_ms,
            "total_ms": total_ms,
            "generated_tokens": tokens_generated,
            "avg_time_per_token_ms": avg_time_per_token_ms
        }
    )
