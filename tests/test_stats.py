import pytest

from ahe.stats import mcnemar_exact, paired_bootstrap


def test_paired_bootstrap_detects_clear_win():
    rng_a = [0.0] * 40 + [1.0] * 10
    rng_b = [1.0] * 50
    res = paired_bootstrap(rng_a, rng_b, n_resamples=2000, seed=42)
    assert res.mean_delta == pytest.approx(0.8)
    assert res.ci_low > 0
    assert res.p_win == 1.0


def test_paired_bootstrap_zero_when_identical():
    a = [0.0, 1.0, 0.5, 1.0]
    res = paired_bootstrap(a, list(a), n_resamples=500, seed=1)
    assert res.mean_delta == 0.0
    assert res.ci_low <= 0 <= res.ci_high


def test_paired_bootstrap_validates_input():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [])
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 0.0], [1.0])


def test_mcnemar_exact_symmetric_when_no_discordance():
    assert mcnemar_exact([True, False], [True, False]) == 1.0


def test_mcnemar_exact_flags_one_sided_domination():
    a = [False] * 12 + [True] * 8
    b = [True] * 20
    p = mcnemar_exact(a, b)
    assert p < 0.01
