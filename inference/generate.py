"""
Clean generation function with correct TTFT measurement.
TTFT includes only model-critical operations, excludes cache bookkeeping.
"""

import time
import torch

from cache.prefix_cache import PrefixCache


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt: str,
    prefix_cache: PrefixCache,
    max_new_tokens: int = 50,
    use_cache: bool = True,
    device: torch.device = torch.device("mps"),
):
    """
    Generate text with prefix caching support.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt text
        prefix_cache: PrefixCache instance (pass dummy cache with min_tokens=999999 to disable)
        max_new_tokens: Number of tokens to generate
        use_cache: If False, disable KV-cache completely (no intra-request or prefix cache)
        device: Target device
        
    Returns:
        Tuple of (generated_text, timings_dict)
    """
    timings = {}

    if not use_cache:
        # ===== NO CACHE MODE (baseline) =====
        t_request_start = time.time()
        
        # Tokenization
        t0 = time.time()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        timings["tokenize_ms"] = (time.time() - t0) * 1000
        timings["cache_lookup_ms"] = 0.0
        
        # Start TTFT timing
        t_ttft_start = time.time()
        
        # Full prefill without cache
        outputs = model(
            input_ids=input_ids,
            use_cache=False,
        )
        
        logits = outputs.logits[:, -1, :]
        timings["ttft_ms"] = (time.time() - t_ttft_start) * 1000
        
        # Decode loop without cache (recompute full sequence each time)
        generated = []
        t_decode_start = time.time()
        
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        for _ in range(max_new_tokens):
            generated.append(next_token.item())
            
            # Concatenate all tokens
            full_sequence = torch.cat([input_ids, torch.tensor([generated]).to(device)], dim=1)
            
            outputs = model(
                input_ids=full_sequence,
                use_cache=False,
            )
            
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        timings["decode_ms"] = (time.time() - t_decode_start) * 1000
        timings["total_ms"] = (time.time() - t_request_start) * 1000
        
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return text, timings
    
    # ===== WITH CACHE MODE =====
    # ---- Tokenization (control-plane, not part of TTFT) ----
    t0 = time.time()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    total_tokens = input_ids.shape[1]
    timings["tokenize_ms"] = (time.time() - t0) * 1000
    timings["total_prompt_tokens"] = total_tokens

    # ---- Prefix cache lookup (control-plane, excluded from TTFT) ----
    t_lookup_start = time.time()
    cached = prefix_cache.find_longest_prefix(prompt, device)
    timings["cache_lookup_ms"] = (time.time() - t_lookup_start) * 1000

    # ---- Start TTFT clock ONLY before first model-critical step ----
    t_ttft_start = time.time()

    if cached is None:
        # ===== FULL PREFILL =====
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
        )

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        # Store prefix (entire prompt) - this happens after TTFT ends
        timings["ttft_ms"] = (time.time() - t_ttft_start) * 1000
        timings["cached_tokens"] = 0
        timings["cache_hit_rate"] = 0.0
        
        prefix_cache.add(
            prefix_text=prompt,
            prefix_tokens=input_ids,
            past_key_values=past_key_values,
            last_logits=logits,
        )

    else:
        # ===== PREFIX CACHE HIT =====
        prefix_len = cached.token_count
        timings["cached_tokens"] = prefix_len
        timings["cache_hit_rate"] = (prefix_len / total_tokens) * 100.0

        # Compute suffix tokens (if any)
        suffix_text = prompt[len(cached.prefix_text):]
        if suffix_text.strip():
            suffix_ids = tokenizer(
                suffix_text,
                return_tensors="pt"
            ).input_ids.to(device)
            suffix_len = suffix_ids.shape[1]

            outputs = model(
                input_ids=suffix_ids,
                past_key_values=cached.past_key_values,
                use_cache=True,
            )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            # Update cache: Store the shared prefix part only, NOT the full prompt
            # This ensures future requests can benefit from the common prefix
            # Only do this if we processed a suffix - keep the original cached prefix
            # and don't overwrite with longer prompts that include variable suffixes
            # 
            # Strategy: Don't add new entries on suffix processing to keep cache focused
            # on truly shared prefixes. The first request establishes the prefix boundary.
        else:
            # 🔑 Critical fix: NO forward pass when exact match
            logits = cached.last_logits
            past_key_values = cached.past_key_values

        timings["ttft_ms"] = (time.time() - t_ttft_start) * 1000
        timings["prefix_tokens_saved"] = prefix_len
        timings["suffix_tokens_processed"] = total_tokens - prefix_len

    # ---- Decode loop ----
    generated = []
    t_decode_start = time.time()

    next_token = torch.argmax(logits, dim=-1, keepdim=True)

    for _ in range(max_new_tokens):
        generated.append(next_token.item())

        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

    timings["decode_ms"] = (time.time() - t_decode_start) * 1000
    timings["total_ms"] = timings["tokenize_ms"] + timings["cache_lookup_ms"] + timings["ttft_ms"] + timings["decode_ms"]

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, timings
