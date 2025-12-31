"""Quick test to verify cache string matching behavior"""

# Simulate what the cache does
cache = {}

# Request 1: Store full prompt
prompt1 = "LONG_PREFIX" + "What is attention?"
cache["key1"] = prompt1

# Request 2: Check for prefix match
prompt2 = "LONG_PREFIX" + "What is backpropagation."

# Does prompt2 start with any cached entry?
for cached_text in cache.values():
    if prompt2.startswith(cached_text):
        print(f"✓ Cache hit! {cached_text}")
        break
else:
    print(f"✗ Cache miss!")
    print(f"  Prompt2: {prompt2}")
    print(f"  Cached:  {cached_text}")
    print(f"  Starts with? {prompt2.startswith(cached_text)}")
