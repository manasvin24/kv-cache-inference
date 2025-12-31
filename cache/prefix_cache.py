"""
Prefix-based KV cache for reusing computation across requests with shared prefixes.

=== CACHE SEMANTICS ===

Key: SHA1 hash of prefix text

Value: CacheEntry containing:
  - past_key_values: KV tensors (stored on CPU)
  - last_logits: Final layer logits
  - prefix_text: Original text for prefix matching
  - prefix_tokens: Tokenized input
  - token_count: Number of tokens
  - created_at: Unix timestamp when entry was created
  - last_accessed: Unix timestamp of last access

Capacity: max_entries (default: 16)
  - When capacity exceeded, evict least recently used (LRU)
  - OrderedDict maintains access order

TTL (Time To Live): max_age_seconds (default: 1200 = 20 minutes)
  - Absolute lifetime: entries expire TTL seconds after creation (NOT idle timeout)
  - TTL resets only on insert, NOT on access (use last_accessed for debugging only)
  - Lazy evaluation: checked on access, not background thread
  - Expired entries treated as cache miss + evicted immediately

Eviction strategy:
  1. LRU: On insert when size > capacity, evict oldest
  2. TTL: On access, if expired, evict immediately
  3. Explicit memory cleanup on eviction (del tensors, optionally empty_cache)

Stats tracked:
  - hits, misses
  - evictions_lru, evictions_ttl
  - current size
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import DynamicCache


@dataclass
class PrefixCacheEntry:
    """Internal cache entry with metadata for LRU and TTL."""
    prefix_text: str
    prefix_tokens: torch.Tensor          # shape: [1, T]
    past_key_values: Tuple                # HF-style tuple (stored on CPU)
    last_logits: torch.Tensor             # shape: [1, vocab]
    token_count: int
    created_at: float                     # Unix timestamp
    last_accessed: float                  # Unix timestamp


class PrefixCache:
    def __init__(
        self, 
        min_tokens: int = 20,
        max_entries: int = 16,
        max_age_seconds: float = 1200.0,  # 20 minutes
    ):
        self.min_tokens = min_tokens
        self.max_entries = max_entries
        self.max_age_seconds = max_age_seconds
        
        # OrderedDict maintains insertion/access order for LRU
        self._entries: OrderedDict[str, PrefixCacheEntry] = OrderedDict()
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions_lru = 0
        self.evictions_ttl = 0
        self.cached_tokens_reused = 0  # Total tokens saved by cache hits

    def _hash(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()
    
    def _evict_entry(self, key: str, reason: str = "lru") -> None:
        """Explicitly evict an entry and free its memory."""
        if key not in self._entries:
            return
        
        entry = self._entries[key]
        
        # Free GPU/CPU memory
        del entry.past_key_values
        del entry.last_logits
        del entry.prefix_tokens
        
        # Remove from cache
        del self._entries[key]
        
        # Track eviction reason
        if reason == "lru":
            self.evictions_lru += 1
        elif reason == "ttl":
            self.evictions_ttl += 1

    def insert(
        self,
        prefix_text: str,
        prefix_tokens: torch.Tensor,
        past_key_values,
        last_logits: torch.Tensor,
    ):
        """Insert a new cache entry. Triggers LRU eviction if capacity exceeded."""
        token_count = prefix_tokens.shape[1]
        if token_count < self.min_tokens:
            return

        # Convert DynamicCache to tuple for storage
        if hasattr(past_key_values, 'to_legacy_cache'):
            pkv_tuple = past_key_values.to_legacy_cache()
        else:
            pkv_tuple = past_key_values

        key = self._hash(prefix_text)
        now = time.time()
        
        # Create new entry with timestamps
        entry = PrefixCacheEntry(
            prefix_text=prefix_text,
            prefix_tokens=prefix_tokens.cpu(),
            past_key_values=self._to_cpu(pkv_tuple),
            last_logits=last_logits.cpu(),
            token_count=token_count,
            created_at=now,
            last_accessed=now,
        )
        
        # Insert into cache (moves to end if already exists)
        self._entries[key] = entry
        self._entries.move_to_end(key)
        
        # LRU eviction: if over capacity, evict oldest
        while len(self._entries) > self.max_entries:
            # First entry is least recently used
            oldest_key = next(iter(self._entries))
            self._evict_entry(oldest_key, reason="lru")

    def lookup_longest_prefix(
        self,
        prompt: str,
        device: torch.device,
    ) -> Optional[PrefixCacheEntry]:
        """
        Longest-prefix match over cached prefixes.
        Implements lazy TTL evaluation (absolute lifetime) and LRU update on access.
        Returns entry with DynamicCache format for transformers 4.57.3+
        """
        now = time.time()
        best = None
        best_len = -1
        best_key = None
        
        # Scan for expired entries (lazy TTL cleanup based on absolute lifetime)
        expired_keys = []
        for key, entry in self._entries.items():
            age = now - entry.created_at  # Absolute age since creation
            if age > self.max_age_seconds:
                expired_keys.append(key)
        
        # Evict expired entries
        for key in expired_keys:
            self._evict_entry(key, reason="ttl")
        
        # Find longest matching prefix among non-expired entries
        for key, entry in self._entries.items():
            if prompt.startswith(entry.prefix_text):
                if entry.token_count > best_len:
                    best = entry
                    best_len = entry.token_count
                    best_key = key

        if best is None:
            self.misses += 1
            return None

        self.hits += 1
        self.cached_tokens_reused += best.token_count  # Track tokens saved
        
        # Update LRU: move to end (most recently used)
        self._entries.move_to_end(best_key)
        best.last_accessed = now  # For debugging/monitoring only (doesn't reset TTL)
        
        # Convert tuple back to DynamicCache for transformers
        pkv_on_device = self._to_device(best.past_key_values, device)
        dynamic_cache = DynamicCache.from_legacy_cache(pkv_on_device)
        
        return PrefixCacheEntry(
            prefix_text=best.prefix_text,
            prefix_tokens=best.prefix_tokens.to(device),
            past_key_values=dynamic_cache,
            last_logits=best.last_logits.to(device),
            token_count=best.token_count,
            created_at=best.created_at,
            last_accessed=best.last_accessed,
        )
    
    def _to_cpu(self, pkv):
        return tuple(
            tuple(t.cpu() for t in layer)
            for layer in pkv
        )

    def _to_device(self, pkv, device):
        return tuple(
            tuple(t.to(device) for t in layer)
            for layer in pkv
        )

    def clear(self):
        """Clear all cached entries and reset stats."""
        self._entries.clear()
        self.hits = 0
        self.misses = 0
        self.evictions_lru = 0
        self.evictions_ttl = 0
        self.cached_tokens_reused = 0

    def get_stats(self) -> dict:
        """Get cache statistics including eviction and token reuse metrics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        total_evictions = self.evictions_lru + self.evictions_ttl

        return {
            "size": len(self._entries),
            "capacity": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total,
            "evictions_lru": self.evictions_lru,
            "evictions_ttl": self.evictions_ttl,
            "total_evictions": total_evictions,
            "cached_tokens_reused": self.cached_tokens_reused,
        }
