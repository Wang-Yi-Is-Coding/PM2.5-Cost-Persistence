"""Draft versions of the main figure panels from the step-1 and step-2 outputs.

The panels show the same quantities as the main figures of the paper in a simple layout; the published
figures were drawn from the same tables with additional styling and map projection.

Usage:  python scripts/make_figures.py --out outputs/example
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
STATE_COLORS = {0: "#d9d9d9", 1: "#c0392b", 2: "#2e86c1", 3: "#7dcea0", 4: "#f5b041"}
STATE_LABELS = {1: "Cost persistence", 2: "Co-decline", 3: "Loss-only decline", 4: "Dual rise"}
GROUPS = ["environment", "demography", "mortality", "valuation", "calibration"]


def fig1(res, fd):
    T = pd.read_csv(os.path.join(res, "trajectories.csv"))
    w = T[T.scope == "World"]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for q, lab, c in (("pm25_idx", "PM2.5 exposure", "#2e86c1"), ("deaths_idx", "Attributable deaths", "#7f8c8d"),
                      ("loss_C_idx", "Monetized loss", "#c0392b")):
        ax.plot(w.year, w[q], label=lab, color=c)
    ax.axhline(100, color="k", lw=0.5)
    ax.set_ylabel("Index (2000 = 100)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "fig1_world_indices.png"), dpi=200)
    plt.close(fig)


def fig2(out, res, fd):
    m = np.load(os.path.join(out, "maps.npz"))
    st = m["state"].reshape(m["shape"])
    cmap = matplotlib.colors.ListedColormap([STATE_COLORS[k] for k in range(5)])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    a1.imshow(st, cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
    a1.set_axis_off()
    for k, lab in STATE_LABELS.items():
        a1.plot([], [], "s", color=STATE_COLORS[k], label=lab)
    a1.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, -0.25), ncol=2)
    S = pd.read_csv(os.path.join(res, "state_shares.csv"))
    scopes = list(dict.fromkeys(S.scope))
    left = np.zeros(len(scopes))
    for s in (1, 2, 3, 4):
        v = S[S.state == s].set_index("scope").reindex(scopes).pop_share.to_numpy()
        a2.barh(scopes, v, left=left, color=STATE_COLORS[s])
        left += v
    a2.invert_yaxis()
    a2.set_xlabel("Share of 2019 population")
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "fig2_states.png"), dpi=200)
    plt.close(fig)


def fig3(res, fd):
    g = pd.read_csv(os.path.join(res, "decomposition_grouped_pct.csv"))
    s = pd.read_csv(os.path.join(res, "decomposition_by_state_pct.csv"))
    s["scope"] = s.state.map(STATE_LABELS)
    d = pd.concat([g, s[g.columns]], ignore_index=True)
    fig, ax = plt.subplots(figsize=(6, 0.45 * len(d) + 1.2))
    y = np.arange(len(d))
    colors = ["#2e86c1", "#8e44ad", "#16a085", "#c0392b", "#bdc3c7"]
    pos_left, neg_left = np.zeros(len(d)), np.zeros(len(d))
    for grp, c in zip(GROUPS, colors):
        v = d[grp + "_pct"].to_numpy()
        left = np.where(v >= 0, pos_left, neg_left)
        ax.barh(y, v, left=left, color=c, label=grp)
        pos_left += np.where(v >= 0, v, 0)
        neg_left += np.where(v < 0, v, 0)
    ax.plot(d.net_pct, y, "kD", ms=4, label="net change")
    ax.set_yticks(y, d.scope)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Contribution (% of 2000 loss)")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "fig3_decomposition.png"), dpi=200)
    plt.close(fig)


def fig4(out, res, fd):
    sy = pd.read_csv(os.path.join(out, "tables", "scenario_country_year.csv")).groupby("year")[["H", "H_cf"]].sum()
    s = pd.read_csv(os.path.join(res, "scenario_by_state.csv"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    a1.plot(sy.index, sy.H / 1e9, "-", color="#c0392b", label="Observed")
    a1.plot(sy.index, sy.H_cf / 1e9, "--", color="#c0392b", label="Exposure fixed at 2011-2013 mean")
    a1.set_ylabel("Monetized loss (US$ billion)")
    a1.legend(frameon=False, fontsize=7)
    y = np.arange(len(s))
    a2.plot(s.loss_change_obs_pct, y, "o", color="k", label="Observed exposure")
    a2.plot(s.loss_change_cf_pct, y, "o", mfc="none", color="k", label="Fixed exposure")
    a2.set_yticks(y, s.state_name)
    a2.invert_yaxis()
    a2.axvline(0, color="k", lw=0.5)
    a2.set_xlabel("Change in loss, 2011-2013 mean to 2019 (%)")
    a2.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "fig4_scenario.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "example"))
    args = ap.parse_args()
    res = os.path.join(args.out, "results")
    fd = os.path.join(args.out, "figures")
    os.makedirs(fd, exist_ok=True)
    fig1(res, fd)
    fig2(args.out, res, fd)
    fig3(res, fd)
    fig4(args.out, res, fd)
    print(f"figures written to {fd}")


if __name__ == "__main__":
    main()
