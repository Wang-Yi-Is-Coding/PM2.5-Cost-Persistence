"""Decomposition of changes in monetized loss.

Seven-factor identity for unit u (a country, or a country-by-state unit) and age group a:

    H_u = sum_a P_u x A_u x p_{a} x M_{a} x r_{u,a} x V x q_{a}

    P    population of the unit
    A    population-weighted cause-weighted attributable fraction, split exactly into
         A_fix = sum_i pop_{i,2000} a_c(PM_{i,t}) / P_2000   (concentration at fixed 2000 weights) and
         R     = (sum_i pop_{i,t} a_c(PM_{i,t}) / P_t) / A_fix  (population redistribution)
    p_a  population share of age group a (national)
    M_a  sum of the age-specific death rates from the five causes (national)
    r    residual calibration term; P A p M r equals the attributable deaths allocated to age group a
    V    income-indexed base VSL
    q_a  age and longevity adjustment, VSL_{c,t,a} / VSL_{c,t}

LMDI-I additive contributions:

    dH_x = sum_a L(H_{a,1}, H_{a,0}) ln(x_{a,1} / x_{a,0}),   L(y, z) = (y - z) / (ln y - ln z)

Cells (unit, age group) with a zero or missing factor at either endpoint are excluded from the
logarithmic terms; their share of the change in loss is reported separately.
"""
import numpy as np
import pandas as pd

FACTORS = {"pop": "P", "pm25": "A_fix", "redistribution": "R", "ageing": "p", "mortality": "M",
           "resid_calib": "r", "vsl_base": "V", "age_value_adj": "q"}
GROUPS = {"environment": ["pm25", "redistribution"], "demography": ["pop", "ageing"],
          "mortality": ["mortality"], "valuation": ["vsl_base", "age_value_adj"], "calibration": ["resid_calib"]}


def logmean(x, y):
    """Logarithmic mean L(x, y); x if x == y; 0 if either value is not positive."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    out = np.zeros(np.broadcast(x, y).shape)
    x, y = np.broadcast_to(x, out.shape), np.broadcast_to(y, out.shape)
    both = (x > 0) & (y > 0)
    eq = both & np.isclose(x, y, rtol=1e-12, atol=0.0)
    out[eq] = x[eq]
    d = both & ~eq
    out[d] = (x[d] - y[d]) / (np.log(x[d]) - np.log(y[d]))
    return out


def factor_panel(units, age_deaths, age_inputs, valuation, national_age_shares=False):
    """Long table of the factors of the seven-factor identity by unit, year and age group.

    units       : unit, iso3, year, P, P2000, E, E_fix, deaths
                  (E = sum_i pop_{i,t} a_c(PM_{i,t}); E_fix uses 2000 population weights)
    age_deaths  : iso3 (or unit), year, age_lower, deaths - attributable deaths by age group
    age_inputs  : iso3, year, age_lower, pshare, M
    valuation   : iso3, year, age_lower, vsl, vsl_age
    national_age_shares : if True, age shares of deaths come from ``age_deaths`` keyed by iso3 and are
                  applied to every unit of the country (used for country-by-state units).
    """
    u = units.copy()
    u["A_fix"] = np.where(u.P2000 > 0, u.E_fix / u.P2000, np.nan)
    u["R"] = np.where((u.P > 0) & (u.A_fix > 0), (u.E / u.P) / u.A_fix, np.nan)
    u["r_unit"] = np.where(u.E > 0, u.deaths / u.E, np.nan)
    key = "iso3" if national_age_shares else "unit"
    ad = age_deaths.copy()
    if key == "unit" and "unit" not in ad.columns:
        ad["unit"] = ad["iso3"]
    tot = ad.groupby([key, "year"]).deaths.transform("sum")
    ad["prop"] = np.where(tot > 0, ad.deaths / tot, np.nan)
    d = u.merge(ad[[key, "year", "age_lower", "prop"]], on=[key, "year"], how="inner")
    d["deaths_age"] = d.deaths * d.prop
    d = d.merge(age_inputs[["iso3", "year", "age_lower", "pshare", "M"]], on=["iso3", "year", "age_lower"], how="left")
    d = d.merge(valuation[["iso3", "year", "age_lower", "vsl", "vsl_age"]], on=["iso3", "year", "age_lower"], how="left")
    d = d.rename(columns={"pshare": "p", "vsl": "V"})
    with np.errstate(invalid="ignore", divide="ignore"):
        d["r"] = d.r_unit * d.prop / (d.p * d.M)
        d["q"] = d.vsl_age / d.V
    d["H_direct"] = d.deaths_age * d.vsl_age
    return d


def lmdi(panel, y0, y1, factors=FACTORS):
    """Additive LMDI-I contributions by (unit, age group) between years y0 and y1.

    Returns (contributions, exclusion summary). Contributions have one column per factor name and
    H0, H1, dH. The exclusion summary gives the share of the total change in loss that falls in
    cells excluded because of zero or missing factors.
    """
    cols = list(factors.values())
    d = panel[panel.year.isin([y0, y1])].replace([np.inf, -np.inf], np.nan)
    all_H = d.groupby("year").H_direct.sum()
    ok = d.dropna(subset=cols + ["H_direct"])
    for c in cols:
        ok = ok[ok[c] > 0]
    n = ok.groupby(["unit", "age_lower"]).year.nunique()
    keep = n[n == 2].index
    ok = ok.set_index(["unit", "age_lower"])
    ok = ok[ok.index.isin(keep)].reset_index()
    ok["H"] = np.prod([ok[c].to_numpy() for c in cols], axis=0)
    w = ok.pivot_table(index=["unit", "age_lower"], columns="year", values=cols + ["H"])
    H0, H1 = w[("H", y0)].to_numpy(), w[("H", y1)].to_numpy()
    L = logmean(H1, H0)
    out = pd.DataFrame(index=w.index)
    for name, c in factors.items():
        out[name] = L * np.log(w[(c, y1)].to_numpy() / w[(c, y0)].to_numpy())
    out["H0"], out["H1"], out["dH"] = H0, H1, H1 - H0
    out = out.reset_index()
    kept = ok.groupby("year").H.sum()
    dH_all = all_H.get(y1, np.nan) - all_H.get(y0, np.nan)
    dH_kept = kept.get(y1, np.nan) - kept.get(y0, np.nan)
    excl = {"dH_all": float(dH_all), "dH_decomposed": float(dH_kept),
            "excluded_share_of_dH": float((dH_all - dH_kept) / dH_all) if dH_all else np.nan,
            "n_cells": int(len(keep))}
    return out, excl


def summarise(contrib, by=None, factors=FACTORS):
    """Sum contributions over age groups (and units) and check closure.

    ``by`` is a column of ``contrib`` (for example 'unit', or a region column added by the caller);
    None gives a single total.
    """
    names = list(factors)
    g = contrib.groupby(by)[names + ["H0", "H1", "dH"]].sum() if by else contrib[names + ["H0", "H1", "dH"]].sum().to_frame().T
    g["sum_of_terms"] = g[names].sum(axis=1)
    g["closure_rel_err"] = (g.sum_of_terms - g.dH).abs() / g.dH.abs()
    return g


def grouped_pct(summary, groups=GROUPS):
    """Grouped contributions as a percentage of the 2000 (start-year) loss of each row."""
    out = pd.DataFrame(index=summary.index)
    out["net_pct"] = 100 * summary.dH / summary.H0
    for k, members in groups.items():
        out[k + "_pct"] = 100 * summary[members].sum(axis=1) / summary.H0
    return out


def pixel_three_terms(H0, H1, D0, D1, a0, a1):
    """Three-term LMDI for single grid cells: exposure, population and mortality, value per death.

    H = D x v with v = H / D. Exposure = L ln(a1/a0); population and mortality = L ln(D1/D0) - exposure;
    value per death = L ln(v1/v0), where L is the logarithmic mean of the cell's loss. Cells with a zero
    endpoint value are left unallocated (all terms zero; mask returned).
    """
    H0, H1, D0, D1, a0, a1 = (np.asarray(v, float) for v in (H0, H1, D0, D1, a0, a1))
    ok = (H0 > 0) & (H1 > 0) & (D0 > 0) & (D1 > 0) & (a0 > 0) & (a1 > 0)
    L = logmean(H1, H0)
    with np.errstate(divide="ignore", invalid="ignore"):
        env = np.where(ok, L * np.log(a1 / a0), 0.0)
        dem = np.where(ok, L * np.log(D1 / D0), 0.0) - env
        val = np.where(ok, L * np.log((H1 / D1) / (H0 / D0)), 0.0)
    return env, dem, val, ok
