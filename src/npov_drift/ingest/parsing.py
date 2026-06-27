"""Wikitext parsing: split a revision into sections and plain text.

Uses ``mwparserfromhell`` to obtain non-overlapping sections (so per-section
word counts sum to the whole) and to strip wiki markup to plain text for later
embedding / stance analysis.
"""

from __future__ import annotations

import mwparserfromhell as mwp
from mwparserfromhell.nodes import Heading

from .. import config
from ..models import Section


def is_boilerplate(heading: str) -> bool:
    return heading.strip().lower() in config.BOILERPLATE_SECTIONS


def is_alt_view_heading(heading: str) -> bool:
    return heading.strip().lower() in config.ALT_VIEW_HEADINGS


def strip_to_text(wikitext: str) -> str:
    """Strip wiki markup to plain text."""
    return mwp.parse(wikitext or "").strip_code().strip()


def _section_body_text(section) -> str:
    """Plain text of a section with its heading line removed."""
    code = mwp.parse(str(section))
    for h in code.filter_headings():
        code.remove(h)
    return code.strip_code().strip()


def parse_sections(wikitext: str) -> list[Section]:
    """Parse wikitext into a flat list of non-overlapping sections.

    The lead (text before the first heading) is returned first with an empty
    heading and level 0. Sub-sections become their own entries (flat=True), so
    the section list is a faithful, non-overlapping partition of the article.
    """
    code = mwp.parse(wikitext or "")
    sections: list[Section] = []
    for sec in code.get_sections(flat=True, include_lead=True, include_headings=True):
        headings = sec.filter_headings()
        if headings:
            h = headings[0]
            heading = h.title.strip_code().strip()
            level = h.level
        else:
            heading = ""
            level = 0
        text = _section_body_text(sec)
        # Skip wholly empty fragments that can appear between adjacent headings.
        if heading == "" and text == "" and not sections:
            # An empty lead still counts as the lead so indexing is stable, but
            # if there is genuinely no content at all we keep it (word_count 0).
            pass
        sections.append(
            Section(
                heading=heading,
                level=level,
                text=text,
                word_count=len(text.split()),
                char_count=len(text),
                is_boilerplate=is_boilerplate(heading),
                is_alt_view=is_alt_view_heading(heading),
            )
        )
    return sections
