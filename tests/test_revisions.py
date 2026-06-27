import pytest

from npov_drift.ingest.revisions import (
    ArticleNotFound,
    _parse_rev_meta,
    build_snapshots,
    fetch_revision_content,
    fetch_revision_metadata,
)


def test_parse_rev_meta_anon_and_hidden():
    r = {
        "revid": 5,
        "parentid": 4,
        "timestamp": "2020-01-01T00:00:00Z",
        "anon": True,
        "user": "1.2.3.4",
        "userid": 0,
        "size": 10,
        "comment": "hi",
        "tags": ["mobile edit"],
    }
    m = _parse_rev_meta(r)
    assert m.anon and m.user == "1.2.3.4" and m.tags == ["mobile edit"]

    hidden = {
        "revid": 6,
        "parentid": 5,
        "timestamp": "2020-01-02T00:00:00Z",
        "userhidden": True,
        "commenthidden": True,
        "size": 12,
    }
    mh = _parse_rev_meta(hidden)
    assert mh.user is None and mh.userid is None and mh.comment is None


def _page(revs, cont=None, **extra):
    out = {"query": {"pages": [{"pageid": 7, "title": "Foo", "revisions": revs, **extra}]}}
    if cont:
        out["continue"] = cont
    return out


def test_fetch_metadata_paginates(client_cls):
    page1 = _page(
        [
            {"revid": 1, "parentid": None, "timestamp": "2020-01-01T00:00:00Z", "user": "a", "userid": 1, "size": 1},
            {"revid": 2, "parentid": 1, "timestamp": "2020-01-02T00:00:00Z", "user": "b", "userid": 2, "size": 2},
        ],
        cont={"rvcontinue": "X"},
    )
    page2 = _page(
        [
            {"revid": 3, "parentid": 2, "timestamp": "2020-01-03T00:00:00Z", "user": "a", "userid": 1, "size": 3},
        ]
    )
    c = client_cls(json_responses=[page1, page2])
    pageid, title, revs = fetch_revision_metadata(c, "Foo")
    assert pageid == 7 and title == "Foo"
    assert [r.revid for r in revs] == [1, 2, 3]
    assert len(c.json_calls) == 2


def test_fetch_metadata_missing_raises(client_cls):
    c = client_cls(json_responses=[{"query": {"pages": [{"missing": True, "title": "Nope"}]}}])
    with pytest.raises(ArticleNotFound):
        fetch_revision_metadata(c, "Nope")


def test_max_revisions_caps_and_stops_paging(client_cls):
    page1 = _page(
        [
            {"revid": 1, "parentid": None, "timestamp": "2020-01-01T00:00:00Z", "user": "a", "userid": 1, "size": 1},
            {"revid": 2, "parentid": 1, "timestamp": "2020-01-02T00:00:00Z", "user": "b", "userid": 2, "size": 2},
        ],
        cont={"rvcontinue": "X"},
    )
    c = client_cls(json_responses=[page1])  # page2 never needed
    pageid, title, revs = fetch_revision_metadata(c, "Foo", max_revisions=1)
    assert [r.revid for r in revs] == [1]
    assert len(c.json_calls) == 1


def test_fetch_content_and_build_snapshots(client_cls):
    resp = {
        "query": {
            "pages": [
                {
                    "pageid": 7,
                    "title": "Foo",
                    "revisions": [
                        {
                            "revid": 10,
                            "timestamp": "2020-05-01T00:00:00Z",
                            "slots": {"main": {"content": "Lead text.\n\n== Criticism ==\nBad stuff here.\n"}},
                        }
                    ],
                }
            ]
        }
    }
    c = client_cls(json_responses=[resp])
    cm = fetch_revision_content(c, [10])
    assert cm[10]["wikitext"].startswith("Lead text.")

    snaps = build_snapshots(cm)
    assert len(snaps) == 1
    s = snaps[0]
    assert s.revid == 10
    headings = [sec.heading for sec in s.sections]
    assert headings == ["", "Criticism"]
    assert any(sec.is_alt_view for sec in s.sections)
