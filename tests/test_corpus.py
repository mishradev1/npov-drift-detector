from npov_drift.reference.corpus import fetch_category_members, select_reference_corpus


def test_fetch_category_members_paginates(client_cls):
    p1 = {"query": {"categorymembers": [{"title": "A"}, {"title": "B"}]}, "continue": {"cmcontinue": "X"}}
    p2 = {"query": {"categorymembers": [{"title": "C"}]}}
    c = client_cls(json_responses=[p1, p2])
    assert fetch_category_members(c, "Category:Featured articles") == ["A", "B", "C"]
    assert len(c.json_calls) == 2


def test_fetch_category_members_respects_max(client_cls):
    p1 = {"query": {"categorymembers": [{"title": "A"}, {"title": "B"}, {"title": "C"}]}, "continue": {"cmcontinue": "X"}}
    c = client_cls(json_responses=[p1])
    assert fetch_category_members(c, "Cat", max_members=2) == ["A", "B"]


def test_select_reference_corpus_caps_and_deterministic(client_cls):
    members = {"query": {"categorymembers": [{"title": t} for t in ["A", "B", "C", "D", "E", "F"]]}}
    bmap = {"A": "politics", "B": "politics", "C": "science", "D": "science", "E": "politics", "F": "science"}

    def fake_classify(_client, title):
        return bmap[title]

    r1 = select_reference_corpus(
        client_cls(json_responses=[members]), ["Cat"], per_bucket=2, seed=7, classify_fn=fake_classify
    )
    r2 = select_reference_corpus(
        client_cls(json_responses=[members]), ["Cat"], per_bucket=2, seed=7, classify_fn=fake_classify
    )
    assert r1 == r2  # deterministic given seed

    counts: dict[str, int] = {}
    for _title, bucket in r1:
        counts[bucket] = counts.get(bucket, 0) + 1
    assert all(v <= 2 for v in counts.values())
    assert set(counts) <= {"politics", "science"}
