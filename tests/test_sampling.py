from npov_drift.ingest.sampling import _evenly_spaced_indices, select_snapshot_revids
from npov_drift.models import RevisionMeta


def rev(revid: int, ts: str) -> RevisionMeta:
    return RevisionMeta(
        revid=revid,
        parentid=None,
        timestamp=ts,
        user="u",
        userid=1,
        anon=False,
        minor=False,
        size=0,
        comment=None,
    )


def test_evenly_spaced_indices():
    assert _evenly_spaced_indices(10, 4) == [0, 3, 6, 9]
    assert _evenly_spaced_indices(5, 10) == [0, 1, 2, 3, 4]
    assert _evenly_spaced_indices(1, 3) == [0]
    assert _evenly_spaced_indices(0, 3) == []


def test_returns_all_when_few():
    revs = [rev(i, f"2020-01-0{i}T00:00:00Z") for i in range(1, 5)]
    assert select_snapshot_revids(revs, max_snapshots=24) == [1, 2, 3, 4]


def test_monthly_picks_last_per_month_plus_endpoints():
    revs = [
        rev(1, "2020-01-01T00:00:00Z"),
        rev(2, "2020-01-15T00:00:00Z"),
        rev(3, "2020-01-20T00:00:00Z"),  # last of Jan
        rev(4, "2020-02-10T00:00:00Z"),  # last of Feb
        rev(5, "2020-03-05T00:00:00Z"),
        rev(6, "2020-03-25T00:00:00Z"),  # last of Mar (also overall last)
    ]
    got = select_snapshot_revids(revs, max_snapshots=4, strategy="monthly")
    # monthly lasts {3,4,6} + endpoints {1,6} -> {1,3,4,6}
    assert got == [1, 3, 4, 6]


def test_evenly_strategy_indices_to_revids():
    revs = [rev(i, f"2020-01-{i:02d}T00:00:00Z") for i in range(1, 11)]  # 10 revs
    got = select_snapshot_revids(revs, max_snapshots=4, strategy="evenly")
    assert got == [1, 4, 7, 10]  # indices 0,3,6,9


def test_empty():
    assert select_snapshot_revids([], max_snapshots=10) == []
