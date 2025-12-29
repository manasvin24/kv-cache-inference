import asyncio
import httpx
import time
import numpy as np

SERVER_URL = "http://localhost:8000/generate"
PROMPT = "Explain KV caching in one sentence."
MAX_NEW_TOKENS = 32
N_CONCURRENT_REQUESTS = 10
N_WARMUP_REQUESTS = 3  # Warmup runs

async def call_api(prompt, use_cache=True, use_prefix_cache=False):
    async with httpx.AsyncClient(timeout=120.0) as client:  # Increased timeout
        start = time.time()
        response = await client.post(
            SERVER_URL,
            json={
                "prompt": prompt,
                "max_new_tokens": MAX_NEW_TOKENS,
                "use_cache": use_cache,
                "use_prefix_cache": use_prefix_cache
            }
        )
        end = time.time()
        latency = end - start
        return latency, response.json()

async def warmup():
    """Warmup runs to initialize model/GPU."""
    print(f"\n{'='*60}")
    print(f"WARMUP: Running {N_WARMUP_REQUESTS} requests...")
    print(f"{'='*60}")
    
    for i in range(N_WARMUP_REQUESTS):
        await call_api(PROMPT, use_cache=True)
        print(f"Warmup {i+1}/{N_WARMUP_REQUESTS} complete")
    
    print("Warmup complete!\n")
    await asyncio.sleep(1)

async def run_benchmark(use_cache=True, use_prefix_cache=False, n_requests=N_CONCURRENT_REQUESTS):
    """Run benchmark with specified cache mode."""
    cache_mode = "WITH KV-cache" if use_cache else "WITHOUT KV-cache"
    if use_prefix_cache:
        cache_mode += " + PREFIX cache"
    
    print(f"\n{'='*60}")
    print(f"Running benchmark: {cache_mode}")
    print(f"Concurrent requests: {n_requests}")
    print(f"{'='*60}")
    
    tasks = [call_api(PROMPT, use_cache=use_cache, use_prefix_cache=use_prefix_cache) for _ in range(n_requests)]
    results = await asyncio.gather(*tasks)

    latencies = [r[0] for r in results]
    responses = [r[1] for r in results]
    
    # Extract server-side metrics
    server_latencies = [r.get('latency_ms', 0) for r in responses]
    tokens_per_sec = [r.get('tokens_per_sec', 0) for r in responses]
    ttft_ms = [r.get('timings', {}).get('ttft_ms', 0) for r in responses]
    decode_ms = [r.get('timings', {}).get('decode_ms', 0) for r in responses]
    cache_memory_mb = [r.get('timings', {}).get('cache_memory_mb', 0) for r in responses]
    
    print(f"\n--- Client-side Latency (includes network) ---")
    print(f"p50 latency: {np.percentile(latencies, 50)*1000:.2f} ms")
    print(f"p95 latency: {np.percentile(latencies, 95)*1000:.2f} ms")
    print(f"Average latency: {np.mean(latencies)*1000:.2f} ms")
    
    print(f"\n--- Server-side Latency (generation only) ---")
    print(f"p50 latency: {np.percentile(server_latencies, 50):.2f} ms")
    print(f"p95 latency: {np.percentile(server_latencies, 95):.2f} ms")
    print(f"Average latency: {np.mean(server_latencies):.2f} ms")
    
    print(f"\n--- TTFT (Time To First Token) ---")
    print(f"Average TTFT: {np.mean(ttft_ms):.2f} ms")
    print(f"p50 TTFT: {np.percentile(ttft_ms, 50):.2f} ms")
    print(f"p95 TTFT: {np.percentile(ttft_ms, 95):.2f} ms")
    
    print(f"\n--- Decode Phase ---")
    print(f"Average decode time: {np.mean(decode_ms):.2f} ms")
    print(f"Average per-token: {np.mean(decode_ms) / MAX_NEW_TOKENS:.2f} ms/token")
    
    print(f"\n--- Memory ---")
    print(f"Average cache memory: {np.mean(cache_memory_mb):.2f} MB")
    
    print(f"\n--- Throughput ---")
    print(f"Average tokens/sec: {np.mean(tokens_per_sec):.2f}")
    print(f"{'='*60}\n")
    
    return {
        'latencies': latencies,
        'server_latencies': server_latencies,
        'tokens_per_sec': tokens_per_sec,
        'ttft_ms': ttft_ms,
        'decode_ms': decode_ms,
        'cache_memory_mb': cache_memory_mb,
        'use_cache': use_cache,
        'use_prefix_cache': use_prefix_cache
    }

async def main():
    """Run comparison benchmark with proper warmup."""
    print("\n" + "="*60)
    print("KV-CACHE INFERENCE BENCHMARK")
    print("="*60)
    
    # STEP 1: Warmup
    await warmup()
    
    # STEP 2: Test WITHOUT cache (to establish baseline)
    results_without_cache = await run_benchmark(
        use_cache=False, 
        use_prefix_cache=False,
        n_requests=N_CONCURRENT_REQUESTS
    )
    
    await asyncio.sleep(2)
    
    # STEP 3: Test WITH KV-cache only
    results_with_cache = await run_benchmark(
        use_cache=True,
        use_prefix_cache=False,
        n_requests=N_CONCURRENT_REQUESTS
    )
    
    await asyncio.sleep(2)
    
    # STEP 4: Test WITH KV-cache + PREFIX cache
    results_with_prefix = await run_benchmark(
        use_cache=True,
        use_prefix_cache=True,
        n_requests=N_CONCURRENT_REQUESTS
    )
    
    # COMPARISON
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    # KV-cache speedup
    kv_speedup = np.mean(results_without_cache['server_latencies']) / np.mean(results_with_cache['server_latencies'])
    kv_ttft_speedup = np.mean(results_without_cache['ttft_ms']) / np.mean(results_with_cache['ttft_ms'])
    kv_decode_speedup = np.mean(results_without_cache['decode_ms']) / np.mean(results_with_cache['decode_ms'])
    
    # Prefix cache speedup
    prefix_ttft_speedup = np.mean(results_with_cache['ttft_ms']) / np.mean(results_with_prefix['ttft_ms'])
    
    print(f"\n--- Baseline (NO cache) ---")
    print(f"Average latency: {np.mean(results_without_cache['server_latencies']):.2f} ms")
    print(f"Average TTFT: {np.mean(results_without_cache['ttft_ms']):.2f} ms")
    print(f"Average decode: {np.mean(results_without_cache['decode_ms']):.2f} ms")
    print(f"Cache memory: {np.mean(results_without_cache['cache_memory_mb']):.2f} MB")
    
    print(f"\n--- WITH KV-cache ---")
    print(f"Average latency: {np.mean(results_with_cache['server_latencies']):.2f} ms")
    print(f"Average TTFT: {np.mean(results_with_cache['ttft_ms']):.2f} ms")
    print(f"Average decode: {np.mean(results_with_cache['decode_ms']):.2f} ms")
    print(f"Cache memory: {np.mean(results_with_cache['cache_memory_mb']):.2f} MB")
    print(f"→ Overall speedup: {kv_speedup:.2f}x")
    print(f"→ TTFT speedup: {kv_ttft_speedup:.2f}x (should be ~1x, warmup is equal)")
    print(f"→ Decode speedup: {kv_decode_speedup:.2f}x (KEY METRIC for KV-cache)")
    
    print(f"\n--- WITH KV-cache + PREFIX cache ---")
    print(f"Average latency: {np.mean(results_with_prefix['server_latencies']):.2f} ms")
    print(f"Average TTFT: {np.mean(results_with_prefix['ttft_ms']):.2f} ms")
    print(f"Average decode: {np.mean(results_with_prefix['decode_ms']):.2f} ms")
    print(f"Cache memory: {np.mean(results_with_prefix['cache_memory_mb']):.2f} MB")
    print(f"→ TTFT speedup vs KV-only: {prefix_ttft_speedup:.2f}x (PREFIX cache benefit)")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    print(f"1. KV-cache improves DECODE phase by {kv_decode_speedup:.2f}x")
    print(f"2. KV-cache TTFT should be ~1x (same prefill work)")
    print(f"3. PREFIX cache improves TTFT by {prefix_ttft_speedup:.2f}x (skips prefill)")
    print(f"4. Memory overhead: {np.mean(results_with_cache['cache_memory_mb']):.2f} MB")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
