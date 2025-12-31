"""
Test LRU and TTL eviction policies.
Demonstrates cache capacity limits and time-based expiration.
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


def test_lru_eviction():
    """Test that LRU eviction works when cache exceeds capacity."""
    print("\n" + "="*80)
    print("TEST 1: LRU EVICTION (capacity=3)")
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
    
    # Small cache capacity to trigger evictions
    cache = PrefixCache(min_tokens=5, max_entries=3, max_age_seconds=3600)
    
    # Create 5 different prefixes (will exceed capacity of 3)
    prefixes = [
        "Prefix A: This is the first test prefix. ",
        "Prefix B: This is the second test prefix. ",
        "Prefix C: This is the third test prefix. ",
        "Prefix D: This is the fourth test prefix. ",
        "Prefix E: This is the fifth test prefix. ",
    ]
    
    print(f"\nAdding {len(prefixes)} prefixes to cache (capacity: 3)...")
    
    for i, prefix in enumerate(prefixes):
        prompt = prefix + "Question?"
        _, timings = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            prefix_cache=cache,
            max_new_tokens=5,
            device=DEVICE
        )
        
        stats = cache.get_stats()
        print(f"\n{i+1}. Added '{prefix[:20]}...'")
        print(f"   Cache size: {stats['size']}/{stats['capacity']}")
        print(f"   LRU evictions: {stats['evictions_lru']}")
    
    # Final stats
    stats = cache.get_stats()
    print(f"\n{'='*80}")
    print(f"Final Cache State:")
    print(f"  Size: {stats['size']}/{stats['capacity']}")
    print(f"  Total evictions: {stats['total_evictions']}")
    print(f"  LRU evictions: {stats['evictions_lru']}")
    print(f"  TTL evictions: {stats['evictions_ttl']}")
    
    # Verify LRU behavior: first 2 prefixes should be evicted
    assert stats['evictions_lru'] == 2, f"Expected 2 LRU evictions, got {stats['evictions_lru']}"
    assert stats['size'] == 3, f"Expected cache size 3, got {stats['size']}"
    
    print(f"\n✅ LRU eviction test passed!")
    

def test_ttl_eviction():
    """Test that TTL eviction works for expired entries."""
    print("\n" + "="*80)
    print("TEST 2: TTL EVICTION (max_age=2 seconds)")
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
    
    # Short TTL to trigger time-based eviction
    cache = PrefixCache(min_tokens=5, max_entries=10, max_age_seconds=2.0)
    
    prefix = "Test prefix for TTL eviction: "
    prompt = prefix + "First question?"
    
    print(f"\n1. Adding entry to cache...")
    _, timings = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        prefix_cache=cache,
        max_new_tokens=5,
        device=DEVICE
    )
    
    stats = cache.get_stats()
    print(f"   Cache size: {stats['size']}")
    print(f"   Hits: {stats['hits']}, Misses: {stats['misses']}")
    
    print(f"\n2. Immediate access (should hit)...")
    _, timings = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        prefix_cache=cache,
        max_new_tokens=5,
        device=DEVICE
    )
    
    stats = cache.get_stats()
    print(f"   Cache size: {stats['size']}")
    print(f"   Hits: {stats['hits']}, Misses: {stats['misses']}")
    
    print(f"\n3. Waiting 2.5 seconds for TTL expiration...")
    time.sleep(2.5)
    
    print(f"\n4. Access after TTL expired (should miss + evict)...")
    _, timings = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        prefix_cache=cache,
        max_new_tokens=5,
        device=DEVICE
    )
    
    stats = cache.get_stats()
    print(f"   Cache size: {stats['size']}")
    print(f"   Hits: {stats['hits']}, Misses: {stats['misses']}")
    print(f"   TTL evictions: {stats['evictions_ttl']}")
    
    # Verify TTL behavior
    assert stats['evictions_ttl'] == 1, f"Expected 1 TTL eviction, got {stats['evictions_ttl']}"
    assert stats['misses'] == 2, f"Expected 2 misses, got {stats['misses']}"
    
    print(f"\n✅ TTL eviction test passed!")


def test_lru_access_order():
    """Test that LRU correctly tracks access order."""
    print("\n" + "="*80)
    print("TEST 3: LRU ACCESS ORDER (capacity=2)")
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
    
    cache = PrefixCache(min_tokens=5, max_entries=2, max_age_seconds=3600)
    
    prompt_a = "Prefix A: First entry. Question?"
    prompt_b = "Prefix B: Second entry. Question?"
    prompt_c = "Prefix C: Third entry. Question?"
    
    print(f"\n1. Add entry A")
    generate(model, tokenizer, prompt_a, cache, max_new_tokens=5, device=DEVICE)
    print(f"   Cache: A")
    
    print(f"\n2. Add entry B")
    generate(model, tokenizer, prompt_b, cache, max_new_tokens=5, device=DEVICE)
    print(f"   Cache: A, B")
    
    print(f"\n3. Access entry A again (refreshes it)")
    generate(model, tokenizer, prompt_a, cache, max_new_tokens=5, device=DEVICE)
    print(f"   Cache: B, A (A is now most recent)")
    
    stats_before = cache.get_stats()
    print(f"   Hits: {stats_before['hits']}, Misses: {stats_before['misses']}")
    
    print(f"\n4. Add entry C (should evict B, not A)")
    generate(model, tokenizer, prompt_c, cache, max_new_tokens=5, device=DEVICE)
    print(f"   Cache: A, C (B was evicted)")
    
    stats = cache.get_stats()
    print(f"\n   Final size: {stats['size']}")
    print(f"   LRU evictions: {stats['evictions_lru']}")
    
    # Verify: B should be evicted, A and C should still be cached
    print(f"\n5. Verify A is still cached...")
    _, timings = generate(model, tokenizer, prompt_a, cache, max_new_tokens=5, device=DEVICE)
    has_cache_hit = timings.get('cached_tokens', 0) > 0
    print(f"   Cache hit: {has_cache_hit}")
    print(f"   Cached tokens: {timings.get('cached_tokens', 0)}")
    
    assert has_cache_hit, "Entry A should still be cached!"
    assert stats['evictions_lru'] == 1, f"Expected 1 eviction, got {stats['evictions_lru']}"
    
    print(f"\n✅ LRU access order test passed!")


def main():
    test_lru_eviction()
    test_ttl_eviction()
    test_lru_access_order()
    
    print("\n" + "="*80)
    print("🎉 ALL EVICTION TESTS PASSED!")
    print("="*80)


if __name__ == "__main__":
    main()
