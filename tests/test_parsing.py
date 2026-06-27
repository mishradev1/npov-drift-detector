from npov_drift.ingest.parsing import (
    is_alt_view_heading,
    is_boilerplate,
    parse_sections,
    strip_to_text,
)

WT = (
    "The subject is a thing.\n"
    "\n"
    "== History ==\n"
    "Some [[history]] text here.\n"
    "\n"
    "== Criticism ==\n"
    "Critics disagree strongly.\n"
    "\n"
    "=== Specific concern ===\n"
    "One nested point.\n"
    "\n"
    "== References ==\n"
    "Citations go here.\n"
)


def test_section_headings_and_levels():
    secs = parse_sections(WT)
    assert [s.heading for s in secs] == [
        "",
        "History",
        "Criticism",
        "Specific concern",
        "References",
    ]
    assert [s.level for s in secs] == [0, 2, 2, 3, 2]


def test_word_counts_hand_computed():
    secs = {s.heading: s for s in parse_sections(WT)}
    assert secs[""].word_count == 5  # "The subject is a thing."
    assert secs["History"].word_count == 4  # link [[history]] -> "history"
    assert secs["Criticism"].word_count == 3
    assert secs["Specific concern"].word_count == 3
    assert secs["References"].word_count == 3


def test_boilerplate_and_alt_view_flags():
    secs = {s.heading: s for s in parse_sections(WT)}
    assert secs["References"].is_boilerplate
    assert not secs["History"].is_boilerplate
    assert secs["Criticism"].is_alt_view
    assert not secs["History"].is_alt_view


def test_strip_markup_links_and_templates():
    assert strip_to_text("Hello [[World]]") == "Hello World"
    assert strip_to_text("A [[B|c]] d") == "A c d"
    # Templates are removed entirely.
    out = strip_to_text("x {{infobox|y=1}} z")
    assert "x" in out and "z" in out and "infobox" not in out


def test_heading_helpers_case_insensitive():
    assert is_boilerplate("REFERENCES")
    assert is_alt_view_heading("Controversy")
    assert not is_boilerplate("History")
