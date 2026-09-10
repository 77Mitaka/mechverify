from examples.literature_check import fourbar_limit_check, slider_crank_check


def test_slider_crank_matches_textbook():
    assert slider_crank_check() < 1e-6


def test_fourbar_limit_matches_law_of_cosines():
    assert fourbar_limit_check() < 1e-3
