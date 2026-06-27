"""Derive the <TOPIC> string used in stance hypotheses.

For Phase 2 the topic is the article's subject (its title). A later phase may
substitute a *central contested claim* instead, which is why callers pass a
topic string explicitly rather than re-deriving it everywhere.
"""

from __future__ import annotations

import re

# Trailing parenthetical disambiguators, e.g. "Mercury (planet)" -> keep, but
# "Foo (disambiguation)" -> drop the noise word. We keep genuine disambiguators
# because they sharpen the topic ("Mercury (element)").
_DISAMBIG_NOISE = {"disambiguation"}


def topic_from_title(title: str) -> str:
    t = title.replace("_", " ").strip()
    m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", t)
    if m and m.group(2).strip().lower() in _DISAMBIG_NOISE:
        return m.group(1).strip()
    return t
