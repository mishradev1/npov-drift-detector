from npov_drift.validation.pov_tags import find_first_tagged_index, has_pov_tag


def test_has_pov_tag():
    assert has_pov_tag("{{POV}}")
    assert has_pov_tag("{{NPOV|date=June 2020}}")
    assert has_pov_tag("lead text {{POV-section}} more")
    assert has_pov_tag("{{ Neutrality }}")
    assert has_pov_tag("{{Unbalanced}}")
    assert not has_pov_tag("{{Infobox person}}")
    assert not has_pov_tag("{{POVray render}}")  # word boundary prevents a false match
    assert not has_pov_tag("")


def test_find_first_tagged_index_monotone():
    assert find_first_tagged_index(10, lambda i: i >= 5) == 5
    assert find_first_tagged_index(10, lambda i: True) == 0
    assert find_first_tagged_index(1, lambda i: True) == 0


def test_find_first_tagged_index_none_when_untagged_at_end():
    assert find_first_tagged_index(10, lambda i: False) is None
    assert find_first_tagged_index(0, lambda i: True) is None
