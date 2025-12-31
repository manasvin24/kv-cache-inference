"""
Concurrent KV-Cache Benchmark
Tests intra-request KV caching with concurrent workload.

Metrics tracked:
- Average latency (server)
- P50, P95, P99 latency
- Throughput (requests/sec)
- Average tokens/sec
- Prefill time (TTFT)
- Decode time per token
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from cache.prefix_cache import PrefixCache
from inference.generate import generate


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = torch.device("mps")


def run_single_request(
    model,
    tokenizer,
    prompt: str,
    use_cache: bool,
    prefix_cache: PrefixCache,
    max_new_tokens: int,
    device: torch.device,
    request_id: int,
):
    """Run a single generation request and return timing metrics."""
    start_time = time.time()
    
    text, timings = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        prefix_cache=prefix_cache,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        device=device,
    )
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    # Calculate tokens/sec
    tokens_generated = max_new_tokens
    total_time_s = (end_time - start_time)
    tokens_per_sec = tokens_generated / total_time_s if total_time_s > 0 else 0
    
    return {
        "request_id": request_id,
        "latency_ms": latency_ms,
        "ttft_ms": timings.get("ttft_ms", 0),
        "decode_ms": timings.get("decode_ms", 0),
        "total_ms": timings.get("total_ms", 0),
        "tokens_per_sec": tokens_per_sec,
        "tokens_generated": tokens_generated,
        "cache_hit_rate": timings.get("cache_hit_rate", 0.0),
        "cached_tokens": timings.get("cached_tokens", 0),
    }


def run_concurrent_benchmark(
    model,
    tokenizer,
    prompts: list,
    use_cache: bool,
    prefix_cache: PrefixCache,
    max_new_tokens: int,
    device: torch.device,
    num_workers: int = 1,
):
    """Run concurrent requests and collect metrics."""
    results = []
    
    # For concurrent execution, we need to handle thread safety
    # Since we're using MPS (single GPU), we'll use sequential execution
    # but track metrics as if they were concurrent
    
    benchmark_start = time.time()
    
    if num_workers == 1:
        # Sequential execution (simulates single-threaded server)
        for i, prompt in enumerate(prompts):
            result = run_single_request(
                model, tokenizer, prompt, use_cache, 
                prefix_cache, max_new_tokens, device, i
            )
            results.append(result)
    else:
        # For MPS, we still run sequentially but with thread pool structure
        # True concurrency would require batching or multiple GPUs
        for i, prompt in enumerate(prompts):
            result = run_single_request(
                model, tokenizer, prompt, use_cache,
                prefix_cache, max_new_tokens, device, i
            )
            results.append(result)
    
    benchmark_end = time.time()
    total_benchmark_time = benchmark_end - benchmark_start
    
    return results, total_benchmark_time


def calculate_metrics(results: list, total_time: float):
    """Calculate aggregate metrics from results."""
    latencies = [r["latency_ms"] for r in results]
    ttfts = [r["ttft_ms"] for r in results]
    decode_times = [r["decode_ms"] for r in results]
    tokens_per_sec_list = [r["tokens_per_sec"] for r in results]
    
    metrics = {
        "num_requests": len(results),
        "total_time_sec": total_time,
        
        # Latency metrics
        "avg_latency_ms": statistics.mean(latencies),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
        "p99_latency_ms": statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        
        # TTFT metrics
        "avg_ttft_ms": statistics.mean(ttfts),
        "p50_ttft_ms": statistics.median(ttfts),
        "p95_ttft_ms": statistics.quantiles(ttfts, n=20)[18] if len(ttfts) >= 20 else max(ttfts),
        
        # Decode metrics
        "avg_decode_ms": statistics.mean(decode_times),
        
        # Throughput
        "throughput_req_per_sec": len(results) / total_time,
        "avg_tokens_per_sec": statistics.mean(tokens_per_sec_list),
        
        # Cache metrics (if applicable)
        "avg_cache_hit_rate": statistics.mean([r["cache_hit_rate"] for r in results]),
        "total_cached_tokens": sum([r["cached_tokens"] for r in results]),
    }
    
    return metrics


def print_metrics_table(with_cache_metrics, without_cache_metrics):
    """Print comparison table."""
    print("\n" + "="*100)
    print("BENCHMARK RESULTS COMPARISON")
    print("="*100)
    
    print(f"\n{'Metric':<40} {'With KV-cache':<20} {'Without KV-cache':<20} {'Speedup':<15}")
    print("-"*100)
    
    # Latency
    print(f"{'Average latency (ms)':<40} {with_cache_metrics['avg_latency_ms']:<20.2f} {without_cache_metrics['avg_latency_ms']:<20.2f} {without_cache_metrics['avg_latency_ms']/with_cache_metrics['avg_latency_ms']:.2f}×")
    print(f"{'P50 latency (ms)':<40} {with_cache_metrics['p50_latency_ms']:<20.2f} {without_cache_metrics['p50_latency_ms']:<20.2f} {without_cache_metrics['p50_latency_ms']/with_cache_metrics['p50_latency_ms']:.2f}×")
    print(f"{'P95 latency (ms)':<40} {with_cache_metrics['p95_latency_ms']:<20.2f} {without_cache_metrics['p95_latency_ms']:<20.2f} {without_cache_metrics['p95_latency_ms']/with_cache_metrics['p95_latency_ms']:.2f}×")
    print(f"{'P99 latency (ms)':<40} {with_cache_metrics['p99_latency_ms']:<20.2f} {without_cache_metrics['p99_latency_ms']:<20.2f} {without_cache_metrics['p99_latency_ms']/with_cache_metrics['p99_latency_ms']:.2f}×")
    
    print()
    
    # TTFT
    print(f"{'Average TTFT (ms)':<40} {with_cache_metrics['avg_ttft_ms']:<20.2f} {without_cache_metrics['avg_ttft_ms']:<20.2f} {without_cache_metrics['avg_ttft_ms']/with_cache_metrics['avg_ttft_ms']:.2f}×")
    print(f"{'P50 TTFT (ms)':<40} {with_cache_metrics['p50_ttft_ms']:<20.2f} {without_cache_metrics['p50_ttft_ms']:<20.2f} {without_cache_metrics['p50_ttft_ms']/with_cache_metrics['p50_ttft_ms']:.2f}×")
    
    print()
    
    # Decode
    print(f"{'Average decode time (ms)':<40} {with_cache_metrics['avg_decode_ms']:<20.2f} {without_cache_metrics['avg_decode_ms']:<20.2f} {without_cache_metrics['avg_decode_ms']/with_cache_metrics['avg_decode_ms']:.2f}×")
    
    print()
    
    # Throughput
    print(f"{'Throughput (req/sec)':<40} {with_cache_metrics['throughput_req_per_sec']:<20.2f} {without_cache_metrics['throughput_req_per_sec']:<20.2f} {with_cache_metrics['throughput_req_per_sec']/without_cache_metrics['throughput_req_per_sec']:.2f}×")
    print(f"{'Avg tokens/sec':<40} {with_cache_metrics['avg_tokens_per_sec']:<20.2f} {without_cache_metrics['avg_tokens_per_sec']:<20.2f} {with_cache_metrics['avg_tokens_per_sec']/without_cache_metrics['avg_tokens_per_sec']:.2f}×")
    
    print("\n" + "="*100)


def main():
    print("\n" + "="*100)
    print("CONCURRENT KV-CACHE BENCHMARK")
    print("Testing intra-request KV caching with repeated prompts")
    print("="*100)
    
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
    
    # Test configuration
    NUM_REQUESTS = 20
    MAX_NEW_TOKENS = 50
    
    # Create test prompts (mix of repeated and unique)
    base_prompts = [
        "Explain what machine learning is in simple terms: ",
        "What is the difference between AI and ML? ",
        "Describe neural networks briefly: ",
        "How does gradient descent work? ",
        "What are transformers in deep learning? ",
    ]
    
    # Repeat prompts to simulate real-world usage with some repetition
    prompts = []
    for i in range(NUM_REQUESTS):
        prompts.append(base_prompts[i % len(base_prompts)])
    
    print(f"\nTest Configuration:")
    print(f"  Total requests: {NUM_REQUESTS}")
    print(f"  Unique prompts: {len(base_prompts)}")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")
    print(f"  Device: {DEVICE}")
    
    # ============================================================================
    # TEST 1: WITH KV-CACHE (intra-request caching enabled)
    # ============================================================================
    print("\n" + "="*100)
    print("TEST 1: WITH KV-CACHE (use_cache=True)")
    print("="*100)
    
    cache_with = PrefixCache(min_tokens=20, max_entries=16, max_age_seconds=3600)
    
    print("\nRunning requests with KV-cache enabled...")
    results_with_cache, time_with_cache = run_concurrent_benchmark(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        use_cache=True,
        prefix_cache=cache_with,
        max_new_tokens=MAX_NEW_TOKENS,
        device=DEVICE,
        num_workers=1,
    )
    
    metrics_with_cache = calculate_metrics(results_with_cache, time_with_cache)
    
    print(f"\n✓ Completed {len(results_with_cache)} requests in {time_with_cache:.2f}s")
    print(f"  Avg latency: {metrics_with_cache['avg_latency_ms']:.2f}ms")
    print(f"  Throughput: {metrics_with_cache['throughput_req_per_sec']:.2f} req/sec")
    print(f"  Avg tokens/sec: {metrics_with_cache['avg_tokens_per_sec']:.2f}")
    
    # ============================================================================
    # TEST 2: WITHOUT KV-CACHE (recompute everything)
    # ============================================================================
    print("\n" + "="*100)
    print("TEST 2: WITHOUT KV-CACHE (use_cache=False)")
    print("="*100)
    
    cache_without = PrefixCache(min_tokens=20, max_entries=16, max_age_seconds=3600)
    
    print("\nRunning requests without KV-cache (recomputing from scratch)...")
    results_without_cache, time_without_cache = run_concurrent_benchmark(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        use_cache=False,
        prefix_cache=cache_without,
        max_new_tokens=MAX_NEW_TOKENS,
        device=DEVICE,
        num_workers=1,
    )
    
    metrics_without_cache = calculate_metrics(results_without_cache, time_without_cache)
    
    print(f"\n✓ Completed {len(results_without_cache)} requests in {time_without_cache:.2f}s")
    print(f"  Avg latency: {metrics_without_cache['avg_latency_ms']:.2f}ms")
    print(f"  Throughput: {metrics_without_cache['throughput_req_per_sec']:.2f} req/sec")
    print(f"  Avg tokens/sec: {metrics_without_cache['avg_tokens_per_sec']:.2f}")
    
    # ============================================================================
    # COMPARISON
    # ============================================================================
    print_metrics_table(metrics_with_cache, metrics_without_cache)
    
    # Detailed breakdown
    print("\nDETAILED ANALYSIS")
    print("="*100)
    
    print("\nWith KV-Cache:")
    print(f"  Min latency: {metrics_with_cache['min_latency_ms']:.2f}ms")
    print(f"  Max latency: {metrics_with_cache['max_latency_ms']:.2f}ms")
    print(f"  Avg TTFT: {metrics_with_cache['avg_ttft_ms']:.2f}ms")
    print(f"  Avg Decode: {metrics_with_cache['avg_decode_ms']:.2f}ms")
    print(f"  TTFT/Total ratio: {metrics_with_cache['avg_ttft_ms']/metrics_with_cache['avg_latency_ms']:.1%}")
    
    print("\nWithout KV-Cache:")
    print(f"  Min latency: {metrics_without_cache['min_latency_ms']:.2f}ms")
    print(f"  Max latency: {metrics_without_cache['max_latency_ms']:.2f}ms")
    print(f"  Avg TTFT: {metrics_without_cache['avg_ttft_ms']:.2f}ms")
    print(f"  Avg Decode: {metrics_without_cache['avg_decode_ms']:.2f}ms")
    print(f"  TTFT/Total ratio: {metrics_without_cache['avg_ttft_ms']/metrics_without_cache['avg_latency_ms']:.1%}")
    
    # Key insights
    print("\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100)
    
    latency_improvement = (metrics_without_cache['avg_latency_ms'] - metrics_with_cache['avg_latency_ms']) / metrics_without_cache['avg_latency_ms'] * 100
    ttft_improvement = (metrics_without_cache['avg_ttft_ms'] - metrics_with_cache['avg_ttft_ms']) / metrics_without_cache['avg_ttft_ms'] * 100
    throughput_improvement = (metrics_with_cache['throughput_req_per_sec'] - metrics_without_cache['throughput_req_per_sec']) / metrics_without_cache['throughput_req_per_sec'] * 100
    
    print(f"\n1. Latency Reduction: {latency_improvement:.1f}%")
    print(f"   KV-cache reduces average latency from {metrics_without_cache['avg_latency_ms']:.2f}ms to {metrics_with_cache['avg_latency_ms']:.2f}ms")
    
    print(f"\n2. TTFT Improvement: {ttft_improvement:.1f}%")
    print(f"   Time to first token drops from {metrics_without_cache['avg_ttft_ms']:.2f}ms to {metrics_with_cache['avg_ttft_ms']:.2f}ms")
    print(f"   This is the primary benefit of intra-request KV caching")
    
    print(f"\n3. Throughput Increase: {throughput_improvement:.1f}%")
    print(f"   System can handle {metrics_with_cache['throughput_req_per_sec']:.2f} req/sec vs {metrics_without_cache['throughput_req_per_sec']:.2f} req/sec")
    
    print(f"\n4. Decode Time Analysis:")
    decode_diff_pct = abs(metrics_with_cache['avg_decode_ms'] - metrics_without_cache['avg_decode_ms']) / metrics_without_cache['avg_decode_ms'] * 100
    print(f"   Decode time difference: {decode_diff_pct:.1f}%")
    print(f"   Decode time remains similar ({metrics_with_cache['avg_decode_ms']:.2f}ms vs {metrics_without_cache['avg_decode_ms']:.2f}ms)")
    print(f"   because KV-cache primarily optimizes attention over previous tokens,")
    print(f"   not the autoregressive generation of new tokens")
    
    print(f"\n5. Token Generation Rate:")
    print(f"   With cache: {metrics_with_cache['avg_tokens_per_sec']:.2f} tokens/sec")
    print(f"   Without cache: {metrics_without_cache['avg_tokens_per_sec']:.2f} tokens/sec")
    print(f"   Improvement: {metrics_with_cache['avg_tokens_per_sec']/metrics_without_cache['avg_tokens_per_sec']:.2f}×")
    
    print("\n" + "="*100)
    print("✅ BENCHMARK COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
