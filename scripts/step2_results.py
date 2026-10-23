"""Step 2: results from the step-1 tables (aggregate series, transition states, decomposition, scenarios).

Writes CSV tables and a summary of key quantities (summary.json) to <out>/results.

Usage:  python scripts/step2_results.py --data data/example --out outputs/example
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
from pmcp import STATE_NAMES  # noqa: E402
from pmcp import decomposition as dc  # noqa: E402
from pmcp import scenarios as sc  # noqa: E402
from pmcp import trends as tr  # noqa: E402

BASE_YEARS = (2011, 2012, 2013)
MIDDLE = ("Upper middle income", "Lower middle income")


def trajectories(panel, countries):
    p = panel.merge(countries[["iso3", "region"]], on="iso3", how="left")
    rows = []
    for name, d in [("World", p)] + [(r, p[p.region == r]) for r in sorted(p.region.dropna().unique())]:
        g = d.groupby("year").apply(lambda x: pd.Series({
            "pop": x["pop"].sum(), "pm25": np.nansum(x.pm25_popw * x["pop"]) / x["pop"][np.isfinite(x.pm25_popw)].sum(),
            "deaths": x.deaths.sum(), "loss_C": x.loss_C.sum(), "loss_A": x.loss_A.sum()}), include_groups=False)
        for q in ("pm25", "deaths", "loss_C"):
            g[q + "_idx"] = tr.index_series(g[q].to_numpy())
        g.insert(0, "scope", name)
        rows.append(g.reset_index())
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "example"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "example"))
    args = ap.parse_args()
    td = os.path.join(args.out, "tables")
    rd_ = os.path.join(args.out, "results")
    os.makedirs(rd_, exist_ok=True)
    rd = lambda f: pd.read_csv(os.path.join(td, f))  # noqa: E731
    countries = pd.read_csv(os.path.join(args.data, "countries.csv"))
    region = dict(zip(countries.iso3, countries.region))
    income = dict(zip(countries.iso3, countries.income_group))
    panel = rd("panel_country_year.csv")
    out = {}

    # ---- 1. aggregate series, peaks and post-peak slopes
    T = trajectories(panel, countries)
    T.to_csv(os.path.join(rd_, "trajectories.csv"), index=False)
    W = T[T.scope == "World"].sort_values("year")
    yrs = W.year.to_numpy()
    pk = tr.peak(W.pm25.to_numpy(), yrs)
    post = W[W.year >= pk["peak_year"]]
    out["world"] = {
        "pm25_peak": pk,
        "endpoint_change_pct": {q: float(100 * (W[q].iloc[-1] / W[q].iloc[0] - 1)) for q in ("pm25", "deaths", "loss_C")},
        "post_peak_change_pct": {q: float(100 * (W[q].iloc[-1] / post[q].iloc[0] - 1)) for q in ("pm25", "deaths", "loss_C")},
        "post_peak_sen_idx_per_yr": {q: tr.sen_slope_ci(post[q + "_idx"].to_numpy(), post.year.to_numpy())
                                     for q in ("pm25", "loss_C")},
    }

    # ---- 2. endpoint quadrants of countries
    p0 = panel[panel.year == 2000].set_index("iso3")
    p1 = panel[panel.year == 2019].set_index("iso3")
    q = pd.DataFrame({"pm_down": p1.pm25_popw < p0.pm25_popw, "loss_down": p1.loss_C < p0.loss_C, "pop2019": p1["pop"]})
    q = q[(p0.loss_C > 0) & (p1.loss_C > 0)]
    q["quadrant"] = np.where(q.pm_down & ~q.loss_down, "exposure down, loss up",
                             np.where(q.pm_down & q.loss_down, "both down",
                                      np.where(~q.pm_down & q.loss_down, "exposure up, loss down", "both up")))
    qs = q.groupby("quadrant").agg(n=("pop2019", "size"), pop2019=("pop2019", "sum"))
    qs["pop_share"] = qs.pop2019 / qs.pop2019.sum()
    qs.to_csv(os.path.join(rd_, "country_quadrants.csv"))

    # ---- 3. transition states
    st = rd("states_country.csv")
    v = st[st.state > 0].copy()
    v["region"], v["income_group"] = v.iso3.map(region), v.iso3.map(income)
    rows = []
    for scope, d in [("World", v)] + [(r, v[v.region == r]) for r in sorted(v.region.dropna().unique())] + \
            [(i, v[v.income_group == i]) for i in sorted(v.income_group.dropna().unique())]:
        tot_p, tot_l = d.pop2019.sum(), d.loss2019.sum()
        for s in (1, 2, 3, 4):
            x = d[d.state == s]
            rows.append({"scope": scope, "state": s, "state_name": STATE_NAMES[s], "pop2019": x.pop2019.sum(),
                         "pop_share": x.pop2019.sum() / tot_p, "loss_share": x.loss2019.sum() / tot_l})
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(rd_, "state_shares.csv"), index=False)
    cp = v[v.state == 1]
    out["cost_persistence"] = {
        "pop2019": float(cp.pop2019.sum()),
        "pop_share": float(cp.pop2019.sum() / v.pop2019.sum()),
        "loss_share": float(cp.loss2019.sum() / v.loss2019.sum()),
        "middle_income_share_of_cp_pop": float(cp[cp.income_group.isin(MIDDLE)].pop2019.sum() / cp.pop2019.sum()),
        "deaths_declining_share_of_cp_pop": float(cp.pop2019_deaths_declining.sum() / cp.pop2019.sum()),
        "valid_pop2019": float(v.pop2019.sum()),
    }

    # ---- 4. decomposition
    fac = rd("factors_country_year.csv").assign(unit=lambda d: d.iso3)
    ages = rd("deaths_country_age_year.csv")
    ai = rd("age_inputs_factors.csv")
    val = rd("valuation_country_year_age.csv")
    panel8 = dc.factor_panel(fac, ages, ai, val)
    con, excl = dc.lmdi(panel8, 2000, 2019)
    con["region"] = con.unit.map(region)
    G = dc.summarise(con)
    R = dc.summarise(con, by="region")
    G.to_csv(os.path.join(rd_, "decomposition_world.csv"), index=False)
    pd.concat([dc.grouped_pct(G).assign(scope="World"), dc.grouped_pct(R).reset_index().rename(columns={"region": "scope"})]) \
        .to_csv(os.path.join(rd_, "decomposition_grouped_pct.csv"), index=False)
    out["decomposition_2000_2019"] = {**{k: float(G[k].iloc[0]) for k in list(dc.FACTORS) + ["dH", "H0", "H1"]},
                                      "closure_rel_err": float(G.closure_rel_err.iloc[0]), "exclusion": excl}
    fs = rd("factors_state_year.csv")
    fs["unit"] = fs.iso3 + "|" + fs.state.astype(str)
    panel_s = dc.factor_panel(fs, ages, ai, val, national_age_shares=True)
    cs, excl_s = dc.lmdi(panel_s, 2000, 2019)
    cs["state"] = cs.unit.str.split("|").str[1].astype(int)
    SS = dc.summarise(cs, by="state")
    dc.grouped_pct(SS).to_csv(os.path.join(rd_, "decomposition_by_state_pct.csv"))
    out["decomposition_by_state_closure_vs_country"] = float(SS.dH.sum() / G.dH.iloc[0] - 1)

    # ---- 5. fixed-exposure scenario and required reductions
    sy = rd("scenario_country_year.csv")
    s19 = sy[sy.year == 2019]
    H19, Hcf19 = s19.H.sum(), s19.H_cf.sum()
    ss_ = rd("scenario_state.csv")
    pos = ss_[ss_.state > 0]
    base_c = panel[panel.year.isin(BASE_YEARS)].groupby("iso3").loss_C.mean()
    kc = rd("kcurve_country.csv")
    ks = np.sort(kc.k.unique())
    curve_w = kc.groupby("k").H_k.sum().reindex(ks).to_numpy()
    rr = sc.required_reductions(ks, curve_w, base_c.sum(), Hcf19)
    Wpm = W.set_index("year").pm25
    obs_rate = sc.annual_change(Wpm.loc[2019], Wpm.loc[list(BASE_YEARS)].mean())
    out["scenario"] = {"H2019_obs": float(H19), "H2019_cf": float(Hcf19), "gap2019": float(Hcf19 - H19),
                       "gap2019_pct_of_obs": float(100 * (Hcf19 / H19 - 1)),
                       "gap_post_mean": float(pos.gap_post.sum()),
                       "cp_share_of_positive_gap": float(pos[pos.state == 1].gap_pos_post.sum() / pos.gap_pos_post.sum()),
                       "world_required": rr, "observed_annual_pm25_change": float(obs_rate)}
    fsb = fs[fs.year.isin(BASE_YEARS)].groupby(["state", "year"])[["loss_C", "pm_x_pop", "P"]].sum().groupby("state").mean()
    f19 = fs[fs.year == 2019].groupby("state")[["loss_C", "pm_x_pop", "P"]].sum()
    kst = rd("kcurve_state.csv").groupby(["state", "k"]).H_k.sum()
    rows = []
    for s in (1, 2, 3, 4):
        if s not in fsb.index:
            continue
        hcf = pos[pos.state == s].Hcf2019.sum()
        h = kst.loc[s].reindex(ks).to_numpy()
        r = sc.required_reductions(ks, h, fsb.loc[s, "loss_C"], hcf)
        rows.append({"state": s, "state_name": STATE_NAMES[s],
                     "pm25_change_pct": 100 * ((f19.loc[s, "pm_x_pop"] / f19.loc[s, "P"]) / (fsb.loc[s, "pm_x_pop"] / fsb.loc[s, "P"]) - 1),
                     "loss_change_obs_pct": 100 * (f19.loc[s, "loss_C"] / fsb.loc[s, "loss_C"] - 1),
                     "loss_change_cf_pct": 100 * (hcf / fsb.loc[s, "loss_C"] - 1), "k_A_pct": 100 * r["k_A"]})
    pd.DataFrame(rows).to_csv(os.path.join(rd_, "scenario_by_state.csv"), index=False)
    caps = rd("caps_country.csv").groupby("cap")[["H_cap", "pop_above_cap"]].sum()
    caps["change_pct"] = 100 * (caps.H_cap / H19 - 1)
    caps.to_csv(os.path.join(rd_, "who_levels_world.csv"))

    # ---- 6. three-term grid-cell decomposition within cost persistence
    pt = rd("pixel_terms_state.csv")
    c1 = pt[pt.state == 1].groupby("deaths_trend")[["env", "dem", "val", "H0", "H1"]].sum()
    (100 * c1[["env", "dem", "val"]].div(c1.H0, axis=0)).assign(net_pct=100 * (c1.H1 / c1.H0 - 1)) \
        .to_csv(os.path.join(rd_, "pixel_terms_cost_persistence_pct.csv"))

    with open(os.path.join(rd_, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
    print(f"step 2 done: results in {rd_}")


if __name__ == "__main__":
    main()
