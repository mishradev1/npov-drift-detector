from npov_drift.stance.sentences import split_sentences


def test_basic_split():
    assert split_sentences("A cat sat. A dog ran. Birds fly!") == [
        "A cat sat.",
        "A dog ran.",
        "Birds fly!",
    ]


def test_abbreviation_does_not_split():
    assert split_sentences("Dr. Smith arrived. He left.") == [
        "Dr. Smith arrived.",
        "He left.",
    ]
    assert split_sentences("Use salt, e.g. NaCl, daily. It helps.") == [
        "Use salt, e.g. NaCl, daily.",
        "It helps.",
    ]
    assert split_sentences("The U.S. economy grew. Prices rose.") == [
        "The U.S. economy grew.",
        "Prices rose.",
    ]


def test_newline_is_hard_boundary():
    assert split_sentences("Line one\nLine two") == ["Line one", "Line two"]


def test_empty_and_whitespace():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_no_terminal_punctuation():
    assert split_sentences("Hello world") == ["Hello world"]
