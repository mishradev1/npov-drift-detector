from npov_drift.ingest.cache import JsonCache, canonical_key


def test_set_get_roundtrip(tmp_path):
    c = JsonCache(tmp_path)
    assert c.get("missing") is None
    assert not c.has("k")
    c.set("k", {"a": 1, "b": [1, 2, 3]})
    assert c.has("k")
    assert c.get("k") == {"a": 1, "b": [1, 2, 3]}


def test_overwrite(tmp_path):
    c = JsonCache(tmp_path)
    c.set("k", 1)
    c.set("k", 2)
    assert c.get("k") == 2


def test_canonical_key_order_invariant():
    assert canonical_key({"a": 1, "b": 2}) == canonical_key({"b": 2, "a": 1})
    assert canonical_key({"a": 1}) != canonical_key({"a": 2})
