"""Attributable fractions and the allocation of attributable deaths to age groups.

Notation follows the Methods:
    AF_k(PM)   attributable fraction for cause k at concentration PM (lookup table, linear interpolation)
    w_{c,k}    country-cause weights, fixed at their 2000-2019 means
    a_c(PM)    = sum_k w_{c,k} AF_k(PM), the cause-weighted attributable fraction
    s_{a}(PM)  age share of attributable deaths in a grid cell
"""
import numpy as np
import pandas as pd

from . import CAUSES


def load_af_table(path):
    """Read an attributable-fraction lookup table.

    The CSV has a column ``pm25`` (µg m-3) and one column ``AF_<cause>`` per cause in ``CAUSES``.
    Returns ``(pm_grid, af_grid)`` with ``af_grid`` of shape (n_causes, n_points).
    """
    t = pd.read_csv(path).sort_values("pm25")
    pm = t["pm25"].to_numpy(float)
    af = np.vstack([t[f"AF_{c}"].to_numpy(float) for c in CAUSES])
    if af.min() < 0 or af.max() > 1:
        raise ValueError("attributable fractions must lie between 0 and 1")
    return pm, af


def af_by_cause(pm, pm_grid, af_grid):
    """AF_k(PM) for every cause: shape (n_causes, n).

    Linear interpolation in concentration; zero below the first table value and the last table value
    above the table. Non-finite concentrations give zero.
    """
    pm = np.asarray(pm, float).ravel()
    ok = np.isfinite(pm)
    x = np.where(ok, pm, 0.0)
    out = np.stack([np.interp(x, pm_grid, af_grid[k], left=0.0, right=af_grid[k][-1])
                    for k in range(af_grid.shape[0])])
    out[:, ~ok] = 0.0
    return out


def cause_weights(rates):
    """Country-cause weights w_{c,k}.

    ``rates`` has columns iso3, year, age_lower and one column per cause (deaths per person-year).
    Each weight is the mean rate of the cause over all years and age groups of the country,
    normalised so that the weights of a country sum to one.
    """
    w = rates.groupby("iso3")[list(CAUSES)].mean()
    s = w.sum(axis=1)
    return w.div(s.where(s > 0), axis=0)


def cause_weighted_af(pm, weights, pm_grid, af_grid):
    """a_c(PM) = sum_k w_{c,k} AF_k(PM).

    ``weights`` is either one weight vector (n_causes,) or one vector per value (n_causes, n).
    """
    af = af_by_cause(pm, pm_grid, af_grid)
    w = np.asarray(weights, float)
    if w.ndim == 1:
        w = w[:, None]
    return (w * af).sum(axis=0)


def age_shares(pm, pop_share, rates, pm_grid, af_grid):
    """Age shares of attributable deaths in each grid cell.

    s_a(PM_i) = p_a sum_k AF_k(PM_i) m_{a,k} / sum_b p_b sum_k AF_k(PM_i) m_{b,k}

    pop_share : (n_ages,) population share of each age group in the country and year
    rates     : (n_ages, n_causes) age- and cause-specific death rates of the country and year
    Returns an array (n_ages, n) whose columns sum to one (zero where no cause has a positive fraction).
    Because the five causes have different attributable-fraction curves, the shares depend on the
    concentration of the grid cell through the cause mix of attributable deaths.
    """
    af = af_by_cause(pm, pm_grid, af_grid)                       # (n_causes, n)
    coef = np.asarray(pop_share, float)[:, None] * np.asarray(rates, float)   # (n_ages, n_causes)
    numer = coef @ af                                            # (n_ages, n)
    den = numer.sum(axis=0)
    out = np.zeros_like(numer)
    ok = den > 0
    out[:, ok] = numer[:, ok] / den[ok]
    return out
