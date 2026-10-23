"""Aggregation of grid-cell arrays to country, state and group tables."""
import numpy as np
import pandas as pd


def group_sum(keys, values, n_keys):
    """Sum ``values`` by integer ``keys`` (0..n_keys-1)."""
    return np.bincount(np.asarray(keys).ravel(), weights=np.asarray(values, float).ravel(), minlength=n_keys)[:n_keys]


def unit_keys(country, state, n_states=5):
    """Integer key for country-by-state units: country_code * n_states + state."""
    return np.asarray(country, np.int64) * n_states + np.asarray(state, np.int64)


def shares(df, value, by, within=None):
    """Share of ``value`` by ``by`` within each ``within`` group (or overall)."""
    if within is None:
        return df.groupby(by)[value].sum() / df[value].sum()
    num = df.groupby([within] + ([by] if isinstance(by, str) else list(by)))[value].sum()
    den = df.groupby(within)[value].sum()
    return num / den.reindex(num.index.get_level_values(0)).to_numpy()


def popw_mean(values, weights):
    """Population-weighted mean ignoring non-finite values."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    ok = np.isfinite(v) & np.isfinite(w)
    return float((v[ok] * w[ok]).sum() / w[ok].sum())


def weighted_median(values, weights):
    """Weighted median (first value whose cumulative weight reaches one half)."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    o = np.argsort(v)
    c = np.cumsum(w[o]) / w.sum()
    return float(v[o][np.searchsorted(c, 0.5)])


def spearman_fisher_ci(x, y, alpha=0.05):
    """Spearman rank correlation with a Fisher-z interval using variance 1.06 / (n - 3)."""
    from scipy.stats import norm, spearmanr
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    rho = float(spearmanr(x[ok], y[ok]).statistic)
    z = np.arctanh(rho)
    se = np.sqrt(1.06 / (n - 3))
    q = norm.ppf(1 - alpha / 2)
    return {"n": n, "rho": rho, "ci_low": float(np.tanh(z - q * se)), "ci_high": float(np.tanh(z + q * se))}


def to_frame(rows):
    return pd.DataFrame(rows)
