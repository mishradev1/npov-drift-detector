"""A tiny atomic gzip-JSON disk cache.

Cache everything: the spec requires the tool be polite to the API. The cache is
keyed by a canonical string describing the request, so re-running any analysis
hits the network zero times once warm.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional


def canonical_key(obj: Any) -> str:
    """Stable string for an arbitrary JSON-able object.

    Sorting keys makes the key invariant to dict ordering so that two
    semantically identical requests map to the same cache entry.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class JsonCache:
    """Stores JSON values as gzip files named by the SHA-1 of their key."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json.gz"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str) -> Optional[Any]:
        path = self._path(key)
        if not path.exists():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        # Atomic write: dump to a temp file in the same dir, then os.replace.
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False)
        os.replace(tmp, path)
