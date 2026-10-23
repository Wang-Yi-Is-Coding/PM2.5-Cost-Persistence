"""Theil-Sen trends, transition states and aggregate series.

Relative Theil-Sen slope: the median of (y_j - y_i)/(j - i) over all year pairs i < j, divided by the
mean of the series, in fraction per year. A series is valid when all years are finite and its mean is
positive.

Transition states (signs of the PM2.5 and loss slopes; a zero slope counts as non-declining):
    1 cost persistence   PM2.5 declining, loss non-declining
    2 co-decline         both declining
    3 loss-only decline  PM2.5 non-declining, loss declining
    4 dual rise          neither declining
    0 not analysed       no valid series
"""
import numpy as np
from scipy.stats import theilslopes


def _pairs(n):
    i, j = np.triu_indices(n, k=1)
    return i, j, (j - i).astype(float)


def theil_sen(y, chunk=200_000):
    """Theil-Sen slope along axis 0 of ``y`` (n_years, n) for annual data; returns (n,) per year."""
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    i, j, lag = _pairs(y.shape[0])
    out = np.empty(y.shape[1])
    for c0 in range(0, y.shape[1], chunk):
        yy = y[:, c0:c0 + chunk]
        out[c0:c0 + chunk] = np.median((yy[j] - yy[i]) / lag[:, None], axis=0)
    return out


def relative_trend(y):
    """Relative Theil-Sen slope (fraction per year) and validity mask for ``y`` (n_years, n)."""
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    finite = np.isfinite(y).all(axis=0)
    mean = np.where(finite, np.nanmean(np.where(np.isfinite(y), y, 0.0), axis=0), np.nan)
    valid = finite & (mean > 0)
    rel = np.full(y.shape[1], np.nan)
    if valid.any():
        rel[valid] = theil_sen(y[:, valid]) / mean[valid]
    return rel, valid


def classify_states(r_pm, r_loss):
    """Transition state of each grid cell from the relative slopes of PM2.5 and loss."""
    r_pm = np.asarray(r_pm, float)
    r_loss = np.asarray(r_loss, float)
    ok = np.isfinite(r_pm) & np.isfinite(r_loss)
    pm_down = r_pm < 0
    loss_down = r_loss < 0
    st = np.zeros(r_pm.shape, np.int8)
    st[ok & pm_down & ~loss_down] = 1
    st[ok & pm_down & loss_down] = 2
    st[ok & ~pm_down & loss_down] = 3
    st[ok & ~pm_down & ~loss_down] = 4
    return st


def declining(r):
    """Trend direction used to split states (for example by attributable deaths): True if r < 0."""
    r = np.asarray(r, float)
    return np.isfinite(r) & (r < 0)


def index_series(values, base_position=0):
    """Index a series to 100 at ``base_position``."""
    v = np.asarray(values, float)
    return 100.0 * v / v[base_position]


def peak(values, years):
    """Year of the raw annual maximum, whether it is interior, and the change from the peak to the last year."""
    v = np.asarray(values, float)
    j = int(np.argmax(v))
    interior = 0 < j < len(v) - 1
    return {"peak_year": int(years[j]), "interior": bool(interior),
            "peak_value": float(v[j]), "change_peak_to_end_pct": 100.0 * (v[-1] / v[j] - 1.0)}


def sen_slope_ci(values, years, alpha=0.95):
    """Theil-Sen slope of an aggregate series with the Sen (1968) confidence interval.

    The interval assumes serially independent years and does not include input uncertainty.
    """
    slope, _, lo, hi = theilslopes(np.asarray(values, float), np.asarray(years, float), alpha=alpha)
    return {"slope": float(slope), "ci_low": float(lo), "ci_high": float(hi)}
