"""Exposure scenarios.

Fixed-exposure scenario: each grid cell's observed loss is rescaled by the ratio of the
cause-weighted attributable fractions at the baseline and observed concentrations,

    H^cf_{i,t} = H_{i,t} a_c(PMbar_{i,base}) / a_c(PM_{i,t}),

with the gap set to zero where the observed attributable fraction is zero. The baseline concentration
is the mean of the baseline years, taken before applying a_c.

Required reduction: every 2019 concentration is scaled by (1 - k) while other inputs are
held fixed,

    H_c(k) = sum_{i in c} H_{i,2019} a_c((1 - k) PM_{i,2019}) / a_c(PM_{i,2019}),

and the reduction that brings H_c(k) to a target is found by linear interpolation on a grid of k.
These are exposure scenarios, not forecasts or evaluations of interventions.
"""
import numpy as np

K_GRID = np.round(np.arange(0.0, 1.0001, 0.02), 2)
WHO_LEVELS = (35.0, 25.0, 15.0, 10.0, 5.0)   # WHO 2021 annual interim targets 1-4 and guideline level


def fixed_exposure(H, a_obs, a_base):
    """Fixed-exposure scenario for arrays of grid cells. Returns (H_cf, gap)."""
    H = np.asarray(H, float)
    a_obs = np.asarray(a_obs, float)
    a_base = np.asarray(a_base, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        H_cf = np.where(a_obs > 0, H * a_base / a_obs, H)
    H_cf = np.where(np.isfinite(H_cf), H_cf, H)
    return H_cf, H_cf - H


def scaled_loss(H, pm, a_obs, af_fn, factors):
    """Loss after scaling concentrations: sum over cells for each factor f of H a(f(PM)) / a(PM).

    ``af_fn(pm)`` returns the cause-weighted attributable fraction of each cell at concentration ``pm``
    (it carries each cell's country weights). ``factors`` is a list of callables applied to ``pm``.
    Only cells with positive loss, finite concentration and positive observed fraction contribute.
    Returns an array (len(factors), n_cells) of rescaled cell losses.
    """
    H = np.asarray(H, float)
    pm = np.asarray(pm, float)
    a_obs = np.asarray(a_obs, float)
    use = np.isfinite(pm) & (H > 0) & (a_obs > 0)
    out = np.zeros((len(factors), H.size))
    for j, f in enumerate(factors):
        out[j, use] = H[use] * af_fn(f(pm))[use] / a_obs[use]
    return out


def k_curve(H, pm, a_obs, af_fn, ks=K_GRID):
    """Rescaled 2019 loss of each cell for every proportional reduction k (rows of the result)."""
    return scaled_loss(H, pm, a_obs, af_fn, [lambda p, k=k: (1.0 - k) * p for k in ks])


def cap_curve(H, pm, a_obs, af_fn, caps=WHO_LEVELS):
    """Rescaled 2019 loss of each cell with concentrations capped at each level."""
    return scaled_loss(H, pm, a_obs, af_fn, [lambda p, c=c: np.minimum(p, c) for c in caps])


def solve_k(ks, h, target):
    """Smallest k on a non-increasing curve h(k) with h(k) = target (linear interpolation).

    Returns 0 if the target is at or above h(0) and 1 if it is at or below h(1).
    """
    ks = np.asarray(ks, float)
    h = np.asarray(h, float)
    if target >= h[0]:
        return 0.0
    if target <= h[-1]:
        return 1.0
    return float(np.interp(-target, -h, ks))


def nonenvironmental_growth(H_cf_end, H_base, years=7):
    """Annual growth of loss with exposure fixed at the baseline: g = (H^cf_2019 / H_base)^(1/years) - 1."""
    return (H_cf_end / H_base) ** (1.0 / years) - 1.0


def annual_change(x_end, x_base, years=7):
    """Annualised proportional change from a baseline value to an end value."""
    return (x_end / x_base) ** (1.0 / years) - 1.0


def required_reductions(ks, h, H_base, H_cf_end, years=7):
    """Reductions for one scope (a country, a state or a sum of country curves).

    k_A: reduction that brings 2019 loss back to its baseline-period level.
    k_B: annual proportional reduction that offsets one year of non-environmental growth g,
         solved from H(k) = H(0) / (1 + g); zero when g is not positive.
    """
    g = nonenvironmental_growth(H_cf_end, H_base, years)
    kA = solve_k(ks, h, H_base)
    kB = solve_k(ks, h, h[0] / (1.0 + g)) if np.isfinite(g) and g > 0 else 0.0
    return {"k_A": kA, "k_B": kB, "g_nonenv": float(g), "H_2019": float(h[0]), "H_base": float(H_base)}
