from fastapi import FastAPI
from pydantic import BaseModel
import time
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from cache.prefix_cache import PrefixCache

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

# Initialize prefix cache (min 5 tokens to allow caching of short prompts)
prefix_cache = PrefixCache(min_prefix_length=5, max_cache_size=100)

#prefix caching will not help much, but intra-request KV caching is doing all the work.

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    use_cache: bool = True  # Toggle KV-cache on/off
    use_prefix_cache: bool = True  # Toggle prefix caching

class GenerateResponse(BaseModel):
    text: str
    latency_ms: float
    tokens_per_sec: float
    timings: dict
    cache_enabled: bool

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    t_request_start = time.time()
    prefix_cache_hit = False
    prefix_tokens_saved = 0

    inputs = tokenizer(req.prompt, return_tensors="pt").to("mps")

    with torch.no_grad():
        if req.use_cache:
            # ---- WITH KV-CACHE ----
            past_key_values = None
            prefix_length = 0
            suffix_input_ids = None
            
            t_prefill_start = time.time()
            
            # Try to use prefix cache if enabled
            if req.use_prefix_cache:
                cached_result = prefix_cache.find_longest_prefix(req.prompt, device="mps")
                if cached_result:
                    prefix_text, prefix_tokens, cached_kv_tuple = cached_result
                    prefix_length = prefix_tokens.shape[-1]
                    prefix_tokens_saved = prefix_length
                    prefix_cache_hit = True
                    
                    # Convert tuple back to DynamicCache if model expects it
                    from transformers import DynamicCache
                    past_key_values = DynamicCache.from_legacy_cache(cached_kv_tuple)
                    
                    # Only process tokens after the cached prefix
                    remaining_prompt = req.prompt[len(prefix_text):]
                    if remaining_prompt:
                        # Tokenize ONLY the suffix (new part)
                        remaining_inputs = tokenizer(remaining_prompt, return_tensors="pt").to("mps")
                        suffix_input_ids = remaining_inputs["input_ids"]
                        
                        # Run model ONLY on suffix with cached KV states
                        outputs = model(
                            input_ids=suffix_input_ids,
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                        logits = outputs.logits
                        past_key_values = outputs.past_key_values
                    else:
                        # Entire prompt was cached, use cached KV directly
                        # We still need to get logits for the first new token
                        # Use a dummy forward pass with the cached states
                        logits = None
                        # past_key_values already set from cache
                else:
                    # No cache hit - run full prefill
                    outputs = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        use_cache=True
                    )
                    logits = outputs.logits
                    past_key_values = outputs.past_key_values
                    
                    # Store prefix in cache immediately after prefill
                    # Convert DynamicCache to tuple if needed
                    if hasattr(past_key_values, 'to_legacy_cache'):
                        kv_tuple = past_key_values.to_legacy_cache()
                    else:
                        kv_tuple = past_key_values
                    prefix_cache.put(req.prompt, inputs["input_ids"], kv_tuple)
            else:
                # Prefix cache disabled - run full prefill
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=True
                )
                logits = outputs.logits
                past_key_values = outputs.past_key_values

            # First token generation
            if logits is not None:
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            else:
                # Generate first token from cached state (entire prompt was cached)
                dummy_input = torch.tensor([[tokenizer.eos_token_id]]).to("mps")
                outputs = model(
                    input_ids=dummy_input,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                logits = outputs.logits
                past_key_values = outputs.past_key_values
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
            
            # Calculate KV cache memory usage
            cache_memory_mb = 0
            if past_key_values:
                # Handle both DynamicCache and tuple formats
                if hasattr(past_key_values, 'to_legacy_cache'):
                    # DynamicCache - convert to tuple
                    kv_tuple = past_key_values.to_legacy_cache()
                else:
                    kv_tuple = past_key_values
                
                for layer_kv in kv_tuple:
                    for kv_tensor in layer_kv:
                        cache_memory_mb += kv_tensor.element_size() * kv_tensor.nelement() / (1024**2)

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
            
            cache_memory_mb = 0  # No cache used

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
    if prefix_cache_hit:
        cache_status += " + PREFIX CACHE HIT"
    
    print(f"\n{'='*50}")
    print(f"[MODE] {cache_status}")
    if prefix_cache_hit:
        print(f"[Prefix Cache] HIT! Skipped prefill for {prefix_tokens_saved} tokens")
        cache_stats = prefix_cache.get_stats()
        print(f"[Prefix Cache Stats] Hit rate: {cache_stats['hit_rate']:.2%} ({cache_stats['hits']}/{cache_stats['total_requests']})")
    print(f"[Prefill] {prefill_ms:.2f} ms" + (" (only suffix)" if prefix_cache_hit else " (full prompt)"))
    print(f"[TTFT] {ttft_ms:.2f} ms (Time To First Token)")
    print(f"[Decode] {decode_ms:.2f} ms")
    print(f"[Tokens] {tokens_generated}")
    print(f"[Per-token] {avg_time_per_token_ms:.2f} ms/token")
    print(f"[Total] {total_ms:.2f} ms")
    print(f"[Throughput] {tokens_per_sec:.2f} tokens/sec")
    if req.use_cache:
        print(f"[Cache Memory] {cache_memory_mb:.2f} MB")
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
            "avg_time_per_token_ms": avg_time_per_token_ms,
            "cache_memory_mb": cache_memory_mb
        }
    )

@app.get("/cache/stats")
def get_cache_stats():
    """Get prefix cache statistics."""
    stats = prefix_cache.get_stats()
    return {
        "cache_size": stats["size"],
        "hits": stats["hits"],
        "misses": stats["misses"],
        "hit_rate": stats["hit_rate"],
        "total_requests": stats["total_requests"]
    }

@app.post("/cache/clear")
def clear_cache():
    """Clear the prefix cache."""
    prefix_cache.clear()
    return {
        "status": "success",
        "message": "Prefix cache cleared"
    }
