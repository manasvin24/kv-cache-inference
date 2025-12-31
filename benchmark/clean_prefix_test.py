"""
Clean benchmark comparing baseline vs prefix cache.
Shows the true benefit of storing last_logits and eliminating redundant forward passes.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from cache.prefix_cache import PrefixCache
from inference.generate import generate


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = torch.device("mps")


def main():
    print("\n" + "="*80)
    print("CLEAN PREFIX CACHE BENCHMARK")
    print("="*80)
    
    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        dtype=torch.float16,
        device_map={"": "mps"},
        low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    
    # Initialize cache with minimum 20 tokens (to ensure caching happens)
    cache = PrefixCache(min_tokens=20)
    
    # Long shared prefix (300+ tokens)
    LONG_PREFIX = """You are an expert AI assistant specializing in machine learning and deep learning. You provide clear, accurate, and detailed explanations of technical concepts. Your responses are educational and include examples when helpful.

When explaining machine learning concepts, you should:
1. Start with a high-level overview that captures the core idea
2. Break down complex topics into digestible components
3. Use analogies and real-world examples to illustrate abstract concepts
4. Provide mathematical formulations when relevant, but explain them intuitively
5. Discuss practical applications and use cases
6. Mention common pitfalls and best practices
7. Reference important research papers or foundational work when appropriate

Your expertise covers:
- Neural networks and deep learning architectures (CNNs, RNNs, Transformers, GANs)
- Natural language processing and large language models
- Computer vision and image processing
- Reinforcement learning and decision-making systems
- Optimization algorithms and training techniques
- Model evaluation, validation, and deployment strategies
- Ethical considerations in AI and machine learning

You communicate in a clear, structured manner. You avoid jargon when simpler terms suffice, but you're not afraid to use technical terminology when it's the most precise way to communicate. You acknowledge uncertainty when appropriate and distinguish between established facts and current research frontiers.

When a user asks a question, you:
- First ensure you understand what they're asking
- Provide a direct answer to their specific question
- Offer additional context that might be helpful
- Suggest related topics they might want to explore
- Encourage follow-up questions for deeper understanding

You stay current with the latest developments in AI and machine learning, and you can discuss both theoretical foundations and practical implementation details. You're comfortable working with popular frameworks like PyTorch, TensorFlow, scikit-learn, and Hugging Face Transformers.

Your goal is to make complex AI concepts accessible while maintaining technical accuracy. You help users build intuition and understanding, not just memorize facts. Now, please answer the following question: """
    
    # Test with different prompts sharing the same prefix
    test_prompts = [
        LONG_PREFIX + "What is attention?",
        LONG_PREFIX + "What is backpropagation.",
        LONG_PREFIX + "What is backpropagation formula?",
        LONG_PREFIX + "What is back propagation in neural networks? Give examples."
    ]
    
    print(f"\nPrompt length: ~{len(test_prompts[0])} characters (~{len(tokenizer(test_prompts[0])['input_ids'])} tokens)")
    print(f"Shared prefix: {len(LONG_PREFIX)} characters")
    print(f"Questions: 4 different questions sharing the same prefix")
    print(f"Max new tokens: 15\n")
    
    # Prime the cache with just the shared prefix
    print("="*80)
    print("PRIMING CACHE WITH SHARED PREFIX")
    print("="*80)
    print("\nProcessing shared prefix...")
    _, prime_timings = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=LONG_PREFIX.rstrip(),  # Cache the prefix alone (without trailing space)
        prefix_cache=cache,
        max_new_tokens=32,  # Generate minimal tokens
        device=DEVICE
    )
    prefix_tokens = prime_timings.get('total_prompt_tokens', 0)
    print(f"Cached {prefix_tokens} prefix tokens")
    print(f"Cache entries: {len(cache._entries)}")
    
    print("\n" + "="*80)
    print("RUNNING BENCHMARK")
    print("="*80)
    
    results = []
    outputs = []
    
    for i, test_prompt in enumerate(test_prompts):
        print(f"\nRequest {i+1}/4: {test_prompt[len(LONG_PREFIX):]}")
        text, timings = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=test_prompt,
            prefix_cache=cache,
            max_new_tokens=32,
            device=DEVICE
        )
        
        results.append(timings)
        outputs.append(text)
        
        # Show key metrics
        total_tokens = timings.get('total_prompt_tokens', 0)
        cached_tokens = timings.get('cached_tokens', 0)
        cache_hit_rate = timings.get('cache_hit_rate', 0.0)
        suffix_tokens = timings.get('suffix_tokens_processed', 0)
        
        # Calculate prefill time (TTFT) and tokens/sec for prefill
        prefill_tokens = suffix_tokens if cached_tokens > 0 else total_tokens
        prefill_time_s = timings['ttft_ms'] / 1000.0
        prefill_tokens_per_sec = prefill_tokens / prefill_time_s if prefill_time_s > 0 else 0
        
        # Calculate decode tokens/sec
        decode_time_s = timings['decode_ms'] / 1000.0
        generated_tokens = 32  # max_new_tokens
        decode_tokens_per_sec = generated_tokens / decode_time_s if decode_time_s > 0 else 0
        
        print(f"  Tokens: {total_tokens} total, {cached_tokens} cached ({cache_hit_rate:.1f}%)")
        print(f"  Tokenize: {timings['tokenize_ms']:.1f}ms")
        print(f"  Cache lookup: {timings['cache_lookup_ms']:.1f}ms")
        print(f"  Prefill: {timings['ttft_ms']:.1f}ms ({prefill_tokens} tokens, {prefill_tokens_per_sec:.1f} tok/s)", end="")
        if "prefix_tokens_saved" in timings:
            print(f" [{cached_tokens} cached + {suffix_tokens} new] ✓")
        else:
            print(" [full prefill]")
        print(f"  Decode: {timings['decode_ms']:.1f}ms ({generated_tokens} tokens, {decode_tokens_per_sec:.1f} tok/s)")
        print(f"  Total: {timings['total_ms']:.1f}ms")
        print(f"  Output: {text}")
        
        time.sleep(0.2)
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Entries: {stats['size']}/{stats['capacity']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Evictions: {stats['total_evictions']} (LRU: {stats['evictions_lru']}, TTL: {stats['evictions_ttl']})")
    
    # Calculate prefill token reuse rate (primary metric)
    total_prompt_tokens = sum(r.get('total_prompt_tokens', 0) for r in results)
    cached_tokens_reused = stats['cached_tokens_reused']
    token_reuse_rate = (cached_tokens_reused / total_prompt_tokens) if total_prompt_tokens > 0 else 0.0
    
    print(f"\n⭐ PRIMARY METRIC - Prefill Token Reuse Rate:")
    print(f"  Cached tokens reused: {cached_tokens_reused}")
    print(f"  Total prompt tokens: {total_prompt_tokens}")
    print(f"  Token reuse rate: {token_reuse_rate:.1%}")
    print(f"  → {cached_tokens_reused} out of {total_prompt_tokens} tokens served from cache")
    
    # Prime request (baseline - full prefix processing)
    print(f"\nPrime request (full prefix prefill):")
    print(f"  Prefill: {prime_timings['ttft_ms']:.1f}ms ({prefix_tokens} tokens)")
    prime_prefill_tok_per_sec = prefix_tokens / (prime_timings['ttft_ms'] / 1000.0) if prime_timings['ttft_ms'] > 0 else 0
    print(f"  Throughput: {prime_prefill_tok_per_sec:.1f} tok/s")
    
    # First request (cache hit after priming)
    first = results[0]
    first_prefill_tokens = first.get('suffix_tokens_processed', first.get('total_prompt_tokens', 0))
    first_prefill_tok_per_sec = first_prefill_tokens / (first['ttft_ms'] / 1000.0) if first['ttft_ms'] > 0 else 0
    
    print(f"\nFirst test request (cache hit):")
    print(f"  Prefill: {first['ttft_ms']:.1f}ms ({first_prefill_tokens} tokens, {first_prefill_tok_per_sec:.1f} tok/s)")
    print(f"  Cache utilization: {first.get('cache_hit_rate', 0):.1f}%")
    
    # All cache hit requests (average)
    avg_ttft_hits = sum(r['ttft_ms'] for r in results) / len(results)
    avg_total_hits = sum(r['total_ms'] for r in results) / len(results)
    avg_cache_util = sum(r.get('cache_hit_rate', 0) for r in results) / len(results)
    
    # Calculate average prefill throughput
    avg_prefill_tokens = sum(r.get('suffix_tokens_processed', 0) for r in results) / len(results)
    avg_prefill_tok_per_sec = avg_prefill_tokens / (avg_ttft_hits / 1000.0) if avg_ttft_hits > 0 else 0
    
    print(f"\nAll cache hit requests (avg):")
    print(f"  Average Prefill: {avg_ttft_hits:.1f}ms ({avg_prefill_tokens:.1f} tokens, {avg_prefill_tok_per_sec:.1f} tok/s)")
    print(f"  Average Total: {avg_total_hits:.1f}ms")
    print(f"  Average Cache utilization: {avg_cache_util:.1f}%")
    
    # CORRECT SPEEDUP: prime_ttft / avg_cache_hit_ttft
    speedup = prime_timings['ttft_ms'] / avg_ttft_hits if avg_ttft_hits > 0 else 0
    print(f"\n🚀 Prefill Speedup: {speedup:.2f}x")
    print(f"   Baseline (full prefix): {prime_timings['ttft_ms']:.1f}ms")
    print(f"   With cache (suffix only): {avg_ttft_hits:.1f}ms")
    
    if speedup > 0:
        saved_ms = prime_timings['ttft_ms'] - avg_ttft_hits
        print(f"   Saved {saved_ms:.1f}ms per request ({saved_ms/prime_timings['ttft_ms']*100:.1f}% reduction)")
    
    # Overall summary
    print(f"\n📊 Overall Performance:")
    all_cache_util = sum(r.get('cache_hit_rate', 0) for r in results) / len(results)
    all_prefill = sum(r['ttft_ms'] for r in results) / len(results)
    all_decode = sum(r['decode_ms'] for r in results) / len(results)
    print(f"  Average cache utilization: {all_cache_util:.1f}%")
    print(f"  Average prefill time: {all_prefill:.1f}ms")
    print(f"  Average decode time: {all_decode:.1f}ms")
    print(f"  Prefill/Decode ratio: {all_prefill/all_decode:.2f}x")



if __name__ == "__main__":
    main()
