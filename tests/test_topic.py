from npov_drift.stance.topic import topic_from_title


def test_underscores_to_spaces():
    assert topic_from_title("Capital_punishment") == "Capital punishment"


def test_keeps_real_disambiguator():
    assert topic_from_title("Mercury (planet)") == "Mercury (planet)"


def test_drops_noise_disambiguator():
    assert topic_from_title("Foo (disambiguation)") == "Foo"
