from npov_drift.ingest.api_client import WikipediaClient
from npov_drift.ingest.cache import JsonCache


def test_caches_and_skips_second_network_call(tmp_path, session_cls, response_cls):
    session = session_cls([response_cls({"hello": "world"})])  # only ONE response queued
    client = WikipediaClient(session=session, cache=JsonCache(tmp_path), min_interval=0)
    d1 = client.get_json({"action": "query"})
    d2 = client.get_json({"action": "query"})  # served from cache, no second pop
    assert d1 == d2 == {"hello": "world"}
    assert sum(1 for c in session.calls if c[0] == "GET") == 1


def test_maxlag_error_triggers_retry(session_cls, response_cls):
    sleeps: list[float] = []
    session = session_cls(
        [
            response_cls({"error": {"code": "maxlag", "info": "lag"}}),
            response_cls({"ok": True}),
        ]
    )
    client = WikipediaClient(
        session=session, cache=None, min_interval=0, sleep=sleeps.append
    )
    assert client.get_json({"action": "query"}) == {"ok": True}
    assert len(sleeps) == 1


def test_retry_after_header_on_503(session_cls, response_cls):
    sleeps: list[float] = []
    session = session_cls(
        [
            response_cls({}, status_code=503, headers={"Retry-After": "2"}),
            response_cls({"ok": True}),
        ]
    )
    client = WikipediaClient(
        session=session, cache=None, min_interval=0, sleep=sleeps.append
    )
    assert client.get_json({"action": "query"}) == {"ok": True}
    assert sleeps == [2.0]
