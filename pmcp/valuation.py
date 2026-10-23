"""Value of a statistical life (VSL) and monetized mortality loss.

    VSL_{c,t}   = VSL_ref (GDPpc_{c,t} / GDPpc_{OECD,t})^e
    VSL_{c,t,a} = VSL_{c,t} (ex_{c,t,a} / LE_{c,t}) omega_a
    H_{i,t}     = sum_a D_{i,t,a} VSL_{c(i),t,a}

Model A uses VSL_{c,t} for every age (age-invariant VSL); model C, the principal model, uses
the age-adjusted VSL. Values are in 2011 purchasing-power-parity US dollars.
"""
import numpy as np
import pandas as pd

VSL_REF = 3.54e6          # reference VSL, 2011 PPP US$ (Yin et al. 2021)
ELASTICITY_HIGH = 0.8     # high-income countries
ELASTICITY_OTHER = 1.2    # all other countries


def income_elasticity(income_group):
    """Income elasticity of VSL: 0.8 for high-income countries and 1.2 elsewhere."""
    if not isinstance(income_group, str):
        return np.nan
    return ELASTICITY_HIGH if income_group.strip().lower() == "high income" else ELASTICITY_OTHER


def fill_series(values, max_extrapolation=5):
    """Fill gaps in an annual series.

    Interior gaps are interpolated linearly; leading and trailing gaps are filled with the nearest
    observed value for at most ``max_extrapolation`` years. Returns (filled values, flags) where flags
    are 'obs', 'interp', 'extrap' or 'missing'.
    """
    s = pd.Series(np.asarray(values, float))
    flag = pd.Series("obs", index=s.index)
    flag[s.isna()] = "missing"
    si = s.interpolate(limit_area="inside")
    flag[s.isna() & si.notna()] = "interp"
    sf = si.ffill(limit=max_extrapolation).bfill(limit=max_extrapolation)
    flag[si.isna() & sf.notna()] = "extrap"
    return sf.to_numpy(), flag.to_numpy()


def base_vsl(gdp_pc, oecd_gdp_pc, elasticity, vsl_ref=VSL_REF):
    """Base VSL. Inputs are same-year GDP per capita values in the same currency units."""
    return vsl_ref * (np.asarray(gdp_pc, float) / np.asarray(oecd_gdp_pc, float)) ** np.asarray(elasticity, float)


def wealth_weight_age(age_lower):
    """Age at which the wealth weight omega is read for an age group starting at ``age_lower``.

    Groups under 20 years take the weight at their age plus 25 years (the parents' age), with all
    groups under 5 years mapped to 25 years; the 20-24 group takes the weight at 25 years; older
    groups take the weight at their own lower bound, capped at 85 years.
    """
    a = int(age_lower)
    base = 0 if a < 5 else min(a, 85)
    return base + 25 if base <= 15 else max(base, 25)


def wealth_weights(age_lowers, points):
    """omega_a for each age group. ``points`` maps age (25, 30, ..., 85) to the published weight."""
    pts = {int(k): float(v) for k, v in dict(points).items()}
    return np.array([pts[wealth_weight_age(a)] for a in age_lowers])


def age_adjusted_vsl(table, points):
    """Age-adjusted VSL (model C) for a long table.

    ``table`` needs columns iso3, year, age_lower, pop (age-specific population), ex (remaining life
    expectancy at the start of the age group) and vsl. Adds LE_mean (population-weighted
    mean of ex over the age groups of the country and year), le_ratio, omega and vsl_age.
    """
    t = table.copy()
    ok = t["pop"].notna() & t["ex"].notna()
    num = (t["ex"] * t["pop"]).where(ok).groupby([t.iso3, t.year]).sum()
    den = t["pop"].where(ok).groupby([t.iso3, t.year]).sum()
    le = (num / den).rename("LE_mean").reset_index()
    t = t.drop(columns=[c for c in ("LE_mean",) if c in t.columns]).merge(le, on=["iso3", "year"], how="left")
    t["le_ratio"] = t["ex"] / t["LE_mean"]
    t["omega"] = wealth_weights(t["age_lower"].to_numpy(), points)
    t["vsl_age"] = t["vsl"] * t["le_ratio"] * t["omega"]
    return t


def collapse_age_groups(table, mapping, value_cols, weight="pop"):
    """Merge age groups by population-weighted averaging of ``value_cols``.

    ``mapping`` maps an input age_lower to an output age_lower (for example {0: 0, 1: 0} merges the
    <1 and 1-4 life-table groups into a single <5 group). Unmapped groups are kept as they are.
    """
    t = table.copy()
    t["age_lower"] = t["age_lower"].map(lambda a: mapping.get(int(a), int(a)))
    w = t[weight].fillna(0.0)
    keys = ["iso3", "year", "age_lower"]
    agg = {c: (t[c] * w).groupby([t[k] for k in keys]).sum() for c in value_cols}
    ws = w.groupby([t[k] for k in keys]).sum()
    out = pd.DataFrame({c: np.where(ws > 0, agg[c] / ws, np.nan) for c in value_cols}, index=ws.index)
    out[weight] = ws
    return out.reset_index()


def loss_per_death(shares, vsl_age):
    """Mean value per death in each grid cell: sum_a s_a VSL_a. ``shares`` is (n_ages, n)."""
    return (np.asarray(shares, float) * np.asarray(vsl_age, float)[:, None]).sum(axis=0)


def monetize(deaths, shares, vsl_age):
    """Monetized loss: H_i = sum_a D_i s_{a,i} VSL_a. Returns (loss, deaths by age (n_ages, n))."""
    d = np.asarray(deaths, float)
    by_age = np.asarray(shares, float) * d[None, :]
    return (by_age * np.asarray(vsl_age, float)[:, None]).sum(axis=0), by_age
