"""Checks of the methods against known values and identities.

Run with ``python -m pytest tests`` or ``python tests/test_methods.py``.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
from pmcp import attributable as at  # noqa: E402
from pmcp import decomposition as dc  # noqa: E402
from pmcp import scenarios as sc  # noqa: E402
from pmcp import trends as tr  # noqa: E402
from pmcp import valuation as va  # noqa: E402

PM_GRID = np.array([1.0, 10.0, 50.0, 100.0])
AF_GRID = np.array([[0.0, 0.10, 0.30, 0.40], [0.0, 0.05, 0.20, 0.35], [0.0, 0.20, 0.40, 0.50],
                    [0.0, 0.15, 0.25, 0.30], [0.0, 0.12, 0.22, 0.28]])


def test_theil_sen_matches_scipy():
    t = np.arange(20.0)
    y = np.column_stack([3 + 0.5 * t + np.sin(t), 10 - 0.2 * t + np.cos(1.3 * t)])
    got = tr.theil_sen(y)
    for j in range(2):
        assert np.isclose(got[j], theilslopes(y[:, j], t)[0])


def test_relative_trend_and_validity():
    y = np.column_stack([np.linspace(10, 8, 20), np.r_[np.nan, np.ones(19)], np.zeros(20)])
    rel, ok = tr.relative_trend(y)
    assert ok.tolist() == [True, False, False]
    assert np.isclose(rel[0], (-2 / 19) / 9.0)


def test_states_zero_slope_counts_as_non_declining():
    st = tr.classify_states([-0.1, -0.1, 0.0, 0.2, 0.0, np.nan], [0.0, -0.2, -0.1, 0.1, 0.0, 0.1])
    assert st.tolist() == [1, 2, 3, 4, 4, 0]


def test_af_interpolation_bounds():
    af = at.af_by_cause([0.5, 1.0, 30.0, 500.0, np.nan], PM_GRID, AF_GRID)
    assert np.allclose(af[:, 0], 0) and np.allclose(af[:, 4], 0)
    assert np.allclose(af[:, 3], AF_GRID[:, -1])
    assert np.isclose(af[0, 2], 0.2)


def test_age_shares_sum_to_one_and_follow_cause_mix():
    p = np.array([0.5, 0.3, 0.2])
    rates = np.array([[1e-4, 1e-4, 5e-4, 1e-4, 1e-4], [3e-4, 2e-4, 1e-4, 6e-4, 5e-4], [9e-4, 5e-4, 4e-4, 2e-3, 2e-3]])
    s = at.age_shares([5.0, 40.0, 90.0], p, rates, PM_GRID, AF_GRID)
    assert np.allclose(s.sum(axis=0), 1.0)
    assert not np.allclose(s[:, 0], s[:, 2])          # shares differ with concentration
    same = at.age_shares([5.0, 90.0], p, rates, PM_GRID, np.repeat(AF_GRID[:1], 5, axis=0))
    assert np.allclose(same[:, 0], same[:, 1])        # identical curves: shares do not depend on PM


def test_base_vsl_and_wealth_mapping():
    assert np.isclose(va.base_vsl(40000, 40000, 1.2), va.VSL_REF)
    assert np.isclose(va.base_vsl(20000, 40000, 1.0), va.VSL_REF / 2)
    assert [va.wealth_weight_age(a) for a in (0, 1, 5, 15, 20, 25, 60, 85, 95)] == [25, 25, 30, 40, 25, 25, 60, 85, 85]
    vals, flags = va.fill_series([np.nan, 1.0, np.nan, 3.0, np.nan])
    assert np.allclose(vals, [1, 1, 2, 3, 3]) and list(flags) == ["extrap", "obs", "interp", "obs", "extrap"]


def test_age_adjusted_vsl_le_mean():
    t = pd.DataFrame({"iso3": "X", "year": 2000, "age_lower": [0, 50], "pop": [3.0, 1.0], "ex": [70.0, 30.0], "vsl": 1e6})
    o = va.age_adjusted_vsl(t, {25: 0.5, 50: 2.0})
    assert np.isclose(o.LE_mean.iloc[0], 60.0)
    assert np.allclose(o.vsl_age, [1e6 * 70 / 60 * 0.5, 1e6 * 30 / 60 * 2.0])


def _toy_panel():
    rows = []
    for u, scale in (("U1", 1.0), ("U2", 2.5)):
        for y, g in ((2000, 0.0), (2019, 1.0)):
            for a, al in enumerate((0, 50, 80)):
                rows.append({"unit": u, "iso3": u, "year": y, "age_lower": al,
                             "P": scale * 1e6 * (1 + 0.3 * g), "A_fix": 0.2 - 0.05 * g, "R": 1 + 0.02 * g,
                             "p": [0.5, 0.3, 0.2][a] * (1 - 0.1 * g * (a == 0)) + 0.05 * g * (a == 2),
                             "M": [1e-4, 1e-3, 5e-3][a] * (1 - 0.2 * g), "r": 1 + 0.1 * a * g,
                             "V": 1e6 * (1 + 0.8 * g) * scale, "q": [1.2, 0.9, 0.5][a] * (1 + 0.05 * g)})
    d = pd.DataFrame(rows)
    d["H_direct"] = np.prod([d[c] for c in dc.FACTORS.values()], axis=0)
    return d


def test_lmdi_closes_exactly():
    con, excl = dc.lmdi(_toy_panel(), 2000, 2019)
    s = dc.summarise(con)
    assert s.closure_rel_err.iloc[0] < 1e-12
    assert excl["excluded_share_of_dH"] == 0


def test_lmdi_excludes_zero_cells():
    d = _toy_panel()
    d.loc[(d.unit == "U2") & (d.age_lower == 80) & (d.year == 2019), "M"] = 0.0
    con, excl = dc.lmdi(d, 2000, 2019)
    assert excl["n_cells"] == 5 and excl["excluded_share_of_dH"] != 0
    assert dc.summarise(con).closure_rel_err.iloc[0] < 1e-12


def test_logmean():
    assert np.isclose(dc.logmean(2.0, 2.0), 2.0)
    assert np.isclose(dc.logmean(np.e, 1.0), (np.e - 1) / 1.0)
    assert dc.logmean(0.0, 1.0) == 0.0


def test_pixel_three_terms_close():
    H0, H1, D0, D1, a0, a1 = 2.0, 3.0, 1.0, 1.2, 0.3, 0.25
    env, dem, val, ok = dc.pixel_three_terms([H0], [H1], [D0], [D1], [a0], [a1])
    assert ok[0] and np.isclose(env[0] + dem[0] + val[0], H1 - H0)


def test_fixed_exposure_and_k_curve():
    H = np.array([10.0, 5.0, 0.0])
    a = np.array([0.2, 0.0, 0.1])
    hcf, gap = sc.fixed_exposure(H, a, np.array([0.3, 0.3, 0.3]))
    assert np.allclose(hcf, [15.0, 5.0, 0.0]) and np.allclose(gap, [5.0, 0.0, 0.0])
    w = np.full((5, 2), 0.2)
    fn = lambda p: at.cause_weighted_af(p, w, PM_GRID, AF_GRID)  # noqa: E731
    pm = np.array([40.0, 80.0])
    Hc = np.array([100.0, 50.0])
    hk = sc.k_curve(Hc, pm, fn(pm), fn).sum(axis=1)
    assert np.isclose(hk[0], 150.0) and np.all(np.diff(hk) <= 1e-12) and np.isclose(hk[-1], 0.0)
    k = sc.solve_k(sc.K_GRID, hk, 120.0)
    assert 0 < k < 1 and np.isclose(np.interp(k, sc.K_GRID, hk), 120.0)
    assert sc.solve_k(sc.K_GRID, hk, 200.0) == 0.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
