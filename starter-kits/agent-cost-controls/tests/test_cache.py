"""Cache correctness: normalisation must produce hits, and never a false hit."""

from __future__ import annotations

from costctl.cache import ResponseCache, cache_key, normalize_prompt


def test_normalisation_collapses_whitespace_case_and_trailing_punctuation() -> None:
    assert normalize_prompt("  How   do REFUNDS work?  ") == "how do refunds work"
    assert normalize_prompt("How do refunds work") == "how do refunds work"
    assert normalize_prompt("how do\n\trefunds work!!") == "how do refunds work"


def test_equivalent_prompts_share_a_key() -> None:
    a = cache_key("How do refunds work?", "gpt-4o-mini")
    b = cache_key("  how do   refunds work  ", "gpt-4o-mini")
    assert a == b


def test_different_content_does_not_collide() -> None:
    a = cache_key("How do refunds work?", "gpt-4o-mini")
    b = cache_key("How do returns work?", "gpt-4o-mini")
    assert a != b


def test_model_and_temperature_are_part_of_the_key() -> None:
    prompt = "How do refunds work?"
    assert cache_key(prompt, "gpt-4o-mini") != cache_key(prompt, "gpt-4o")
    assert cache_key(prompt, "gpt-4o-mini", temperature=0.0) != cache_key(
        prompt, "gpt-4o-mini", temperature=0.5
    )


def test_extra_namespace_separates_tenants() -> None:
    prompt = "How do refunds work?"
    assert cache_key(prompt, "gpt-4o-mini", extra="tenant-a") != cache_key(
        prompt, "gpt-4o-mini", extra="tenant-b"
    )


def test_keys_are_stable_across_processes() -> None:
    # SHA-256 rather than the salted built-in hash, so a restart does not empty a
    # shared cache.
    assert cache_key("stable", "gpt-4o-mini") == cache_key("stable", "gpt-4o-mini")
    assert len(cache_key("stable", "gpt-4o-mini")) == 64


def test_miss_then_hit_on_a_normalized_variant() -> None:
    cache = ResponseCache()
    calls = {"n": 0}

    def factory() -> str:
        calls["n"] += 1
        return "cached-answer"

    first, was_cached = cache.get_or_call("How do refunds work?", "gpt-4o-mini", factory)
    assert (first, was_cached) == ("cached-answer", False)
    assert calls["n"] == 1

    second, was_cached = cache.get_or_call("  how do   REFUNDS work  ", "gpt-4o-mini", factory)
    assert (second, was_cached) == ("cached-answer", True)
    assert calls["n"] == 1  # the factory was not called again

    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.stats.hit_rate == 0.5


def test_different_model_is_a_miss() -> None:
    cache = ResponseCache()
    cache.get_or_call("q", "gpt-4o-mini", lambda: "small")
    value, was_cached = cache.get_or_call("q", "gpt-4o", lambda: "large")
    assert (value, was_cached) == ("large", False)


def test_high_temperature_responses_are_not_cached() -> None:
    cache = ResponseCache(max_cacheable_temperature=0.2)
    calls = {"n": 0}

    def factory() -> int:
        calls["n"] += 1
        return calls["n"]

    for _ in range(3):
        _, was_cached = cache.get_or_call("q", "gpt-4o-mini", factory, temperature=0.9)
        assert was_cached is False
    assert calls["n"] == 3
    assert len(cache) == 0


def test_entries_expire_after_the_ttl() -> None:
    now = [0.0]
    cache = ResponseCache(ttl_s=10.0, clock=lambda: now[0])
    key = cache_key("q", "gpt-4o-mini")
    cache.set(key, "value")

    now[0] = 9.9
    assert cache.get(key) == "value"

    now[0] = 10.0
    assert cache.get(key) is None
    assert cache.stats.expirations == 1


def test_lru_eviction_is_bounded_and_keeps_recently_used_entries() -> None:
    cache = ResponseCache(max_entries=2, ttl_s=None)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # "a" is now the most recently used
    cache.set("c", 3)  # evicts "b"

    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3
    assert cache.stats.evictions == 1
    assert len(cache) == 2


def test_a_failing_factory_is_not_cached() -> None:
    cache = ResponseCache()
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient upstream failure")
        return "recovered"

    try:
        cache.get_or_call("q", "gpt-4o-mini", flaky)
    except RuntimeError:
        pass

    value, was_cached = cache.get_or_call("q", "gpt-4o-mini", flaky)
    assert (value, was_cached) == ("recovered", False)


def test_clear_drops_entries_but_keeps_counters() -> None:
    cache = ResponseCache()
    cache.set("k", "v")
    cache.get("k")
    cache.clear()
    assert len(cache) == 0
    assert cache.stats.hits == 1


def test_unicode_variants_normalise_together() -> None:
    # NFKC folds the full-width characters onto their ASCII equivalents.
    assert normalize_prompt("ＲＥＦＵＮＤ") == normalize_prompt("refund")


def test_cache_is_safe_under_concurrent_use() -> None:
    import threading

    cache = ResponseCache(max_entries=64, ttl_s=None)

    def worker(offset: int) -> None:
        for i in range(200):
            key = cache_key(f"prompt {(i + offset) % 32}", "gpt-4o-mini")
            if cache.get(key) is None:
                cache.set(key, i)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(cache) <= 64
    assert cache.stats.lookups == 800
