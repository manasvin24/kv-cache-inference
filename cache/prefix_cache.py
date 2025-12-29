"""Prefix-based KV cache for reusing computation across requests with shared prefixes."""

import hashlib
import torch
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class CachedPrefix:
    """Cached KV states for a prompt prefix."""
    prefix_text: str
    prefix_tokens: torch.Tensor
    past_key_values: Tuple  # Stored on CPU
    num_tokens: int


class PrefixCache:
    """Cache for storing and retrieving KV states of prompt prefixes."""
    
    def __init__(self, min_prefix_length: int = 10, max_cache_size: int = 100):
        """
        Initialize the prefix cache.
        
        Args:
            min_prefix_length: Minimum number of tokens to cache a prefix
            max_cache_size: Maximum number of prefixes to store
        """
        self.min_prefix_length = min_prefix_length
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, CachedPrefix] = {}
        self.hits = 0
        self.misses = 0
    
    def _compute_hash(self, prefix_text: str) -> str:
        """Compute hash key for a prefix."""
        return hashlib.sha256(prefix_text.encode()).hexdigest()
    
    def _move_kv_to_cpu(self, past_key_values: Tuple) -> Tuple:
        """Move KV cache from device to CPU for storage."""
        cpu_kv = []
        for layer_kv in past_key_values:
            cpu_layer = tuple(kv.cpu() for kv in layer_kv)
            cpu_kv.append(cpu_layer)
        return tuple(cpu_kv)
    
    def _move_kv_to_device(self, past_key_values: Tuple, device: str) -> Tuple:
        """Move KV cache from CPU to target device."""
        device_kv = []
        for layer_kv in past_key_values:
            device_layer = tuple(kv.to(device) for kv in layer_kv)
            device_kv.append(device_layer)
        return tuple(device_kv)
    
    def put(self, prefix_text: str, prefix_tokens: torch.Tensor, past_key_values: Tuple):
        """
        Store a prefix and its KV cache.
        
        Args:
            prefix_text: The text of the prefix
            prefix_tokens: Token IDs of the prefix
            past_key_values: KV cache tensors from the model
        """
        num_tokens = prefix_tokens.shape[-1]
        
        # Only cache if prefix is long enough
        if num_tokens < self.min_prefix_length:
            return
        
        # Evict oldest entry if cache is full
        if len(self.cache) >= self.max_cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        # Move to CPU for storage
        cpu_kv = self._move_kv_to_cpu(past_key_values)
        
        cache_key = self._compute_hash(prefix_text)
        self.cache[cache_key] = CachedPrefix(
            prefix_text=prefix_text,
            prefix_tokens=prefix_tokens.cpu(),
            past_key_values=cpu_kv,
            num_tokens=num_tokens
        )
    
    def get(self, prefix_text: str, device: str = "mps") -> Optional[Tuple[torch.Tensor, Tuple]]:
        """
        Retrieve cached KV states for a prefix.
        
        Args:
            prefix_text: The prefix to look up
            device: Target device to move cache to (e.g., "mps", "cuda", "cpu")
            
        Returns:
            Tuple of (prefix_tokens, past_key_values) if found, None otherwise
        """
        cache_key = self._compute_hash(prefix_text)
        
        if cache_key not in self.cache:
            self.misses += 1
            return None
        
        self.hits += 1
        cached = self.cache[cache_key]
        
        # Move from CPU to target device
        device_kv = self._move_kv_to_device(cached.past_key_values, device)
        device_tokens = cached.prefix_tokens.to(device)
        
        return device_tokens, device_kv
    
    def find_longest_prefix(self, prompt: str, device: str = "mps") -> Optional[Tuple[str, torch.Tensor, Tuple]]:
        """
        Find the longest cached prefix that matches the start of the prompt.
        
        Args:
            prompt: The full prompt to match against
            device: Target device for the cached tensors
            
        Returns:
            Tuple of (matched_prefix_text, prefix_tokens, past_key_values) if found, None otherwise
        """
        longest_match = None
        longest_length = 0
        
        for cached_prefix in self.cache.values():
            if prompt.startswith(cached_prefix.prefix_text):
                if cached_prefix.num_tokens > longest_length:
                    longest_match = cached_prefix
                    longest_length = cached_prefix.num_tokens
        
        if longest_match:
            self.hits += 1
            device_kv = self._move_kv_to_device(longest_match.past_key_values, device)
            device_tokens = longest_match.prefix_tokens.to(device)
            return longest_match.prefix_text, device_tokens, device_kv
        
        self.misses += 1
        return None
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total
        }
