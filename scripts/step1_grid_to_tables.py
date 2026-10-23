"""Step 1: from grid-cell arrays to the country, state and scenario tables used in step 2.

Inputs (see data/README.md for the formats):
    countries.csv, gdp_pc.csv, age_inputs.csv, wealth_weights.csv, af_table.csv, grid.npz
Outputs (in <out>/tables and <out>/maps.npz):
    valuation_country_year_age.csv  base and age-adjusted VSL
    panel_country_year.csv          population, population-weighted PM2.5, deaths and loss (models A and C)
    factors_country_year.csv        population and exposure sums for the decomposition
    deaths_country_age_year.csv     attributable deaths by age group
    age_inputs_factors.csv          population shares p_a and summed death rates M_a
    states_country.csv              2019 population and loss by country and transition state
    factors_state_year.csv          the sums of factors_country_year by country and transition state
    scenario_country_year.csv       observed and fixed-exposure loss, 2013-2019
    scenario_state.csv              the same by state, with the 2017-2019 mean gap
    kcurve_country.csv, kcurve_state.csv, caps_country.csv   proportional reductions and WHO levels
    pixel_terms_state.csv           three-term grid-cell decomposition summed by country and state

Usage:  python scripts/step1_grid_to_tables.py --data data/example --out outputs/example
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
from pmcp import CAUSES  # noqa: E402
from pmcp import attributable as at  # noqa: E402
from pmcp import decomposition as dc  # noqa: E402
from pmcp import scenarios as sc  # noqa: E402
from pmcp import trends as tr  # noqa: E402
from pmcp import valuation as va  # noqa: E402

BASE_YEARS = (2011, 2012, 2013)
POST_YEARS = (2017, 2018, 2019)
NS = 5   # states 0-4


def valuation_table(countries, gdp, ai, ww):
    rows = []
    for i3, s in gdp.groupby("iso3"):
        s = s.sort_values("year")
        filled, flag = va.fill_series(s.gdp_pc.to_numpy())
        rows.append(pd.DataFrame({"iso3": i3, "year": s.year.to_numpy(), "gdp_pc": filled, "gdp_flag": flag,
                                  "oecd_gdp_pc": s.oecd_gdp_pc.to_numpy()}))
    v = pd.concat(rows).merge(countries[["iso3", "income_group"]], on="iso3", how="left")
    v["e"] = v.income_group.map(va.income_elasticity)
    v["vsl"] = va.base_vsl(v.gdp_pc, v.oecd_gdp_pc, v.e)
    t = ai[["iso3", "year", "age_lower", "pshare", "ex"]].rename(columns={"pshare": "pop"})
    t = t.merge(v[["iso3", "year", "vsl"]], on=["iso3", "year"], how="left")
    points = dict(zip(ww.age.astype(int), ww.omega.astype(float)))
    t = va.age_adjusted_vsl(t, points).rename(columns={"pop": "pshare"})
    return v, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "example"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "example"))
    args = ap.parse_args()
    td = os.path.join(args.out, "tables")
    os.makedirs(td, exist_ok=True)
    rd = lambda f: pd.read_csv(os.path.join(args.data, f))  # noqa: E731
    countries, gdp, ai, ww = rd("countries.csv"), rd("gdp_pc.csv"), rd("age_inputs.csv"), rd("wealth_weights.csv")
    pm_grid, af_grid = at.load_af_table(os.path.join(args.data, "af_table.csv"))
    g = np.load(os.path.join(args.data, "grid.npz"))
    years = g["years"].astype(int)
    T = len(years)
    yi = {int(y): k for k, y in enumerate(years)}
    code = g["country"].ravel().astype(np.int64)
    iso_of = [str(x) for x in g["country_iso3"]]
    pm = g["pm25"].reshape(T, -1).astype(float)
    pop = g["pop"].reshape(T, -1).astype(float)
    deaths = g["deaths"].reshape(T, -1).astype(float)
    pop = np.where(np.isfinite(pop) & (pop > 0), pop, 0.0)
    ncode = len(iso_of)
    land = code > 0

    # ---- valuation
    vtab, vage = valuation_table(countries, gdp, ai, ww)
    vage.to_csv(os.path.join(td, "valuation_country_year_age.csv"), index=False)
    ages = sorted(ai.age_lower.unique())
    A = len(ages)

    # ---- cause weights and cause-weighted attributable fraction of every cell and year
    w = at.cause_weights(ai)
    W = np.full((len(CAUSES), ncode), 1.0 / len(CAUSES))
    for c in range(1, ncode):
        if iso_of[c] in w.index:
            W[:, c] = w.loc[iso_of[c]].to_numpy()
    Wpix = W[:, code]
    a = np.zeros((T, code.size))
    for k in range(T):
        a[k] = at.cause_weighted_af(pm[k], Wpix, pm_grid, af_grid)
    a[:, ~land] = 0.0
    af_fn = lambda p: at.cause_weighted_af(p, Wpix, pm_grid, af_grid)  # noqa: E731

    # ---- age allocation and monetized loss
    H_C = np.full((T, code.size), np.nan)
    H_A = np.full((T, code.size), np.nan)
    dage = np.zeros((T, ncode, A))
    for c in range(1, ncode):
        m = (code == c)
        for y, k in yi.items():
            s = ai[(ai.iso3 == iso_of[c]) & (ai.year == y)].set_index("age_lower").reindex(ages)
            vv = vage[(vage.iso3 == iso_of[c]) & (vage.year == y)].set_index("age_lower").reindex(ages)
            if s.empty or vv.vsl_age.isna().any():
                continue                      # countries without valuation inputs are excluded
            d = np.where(np.isfinite(deaths[k][m]), deaths[k][m], 0.0)
            sh = at.age_shares(pm[k][m], s.pshare.to_numpy(), s[list(CAUSES)].to_numpy(), pm_grid, af_grid)
            H, by_age = va.monetize(d, sh, vv.vsl_age.to_numpy())
            H_C[k][m] = H
            H_A[k][m] = d * float(vv.vsl.iloc[0])
            dage[k, c] = by_age.sum(axis=1)

    # ---- country-year panel and exposure sums
    rows, frows, arows = [], [], []
    for y, k in yi.items():
        pmc = np.where(np.isfinite(pm[k]), pm[k], 0.0)
        P = np.bincount(code, weights=pop[k], minlength=ncode)
        P0 = np.bincount(code, weights=pop[0], minlength=ncode)
        E = np.bincount(code, weights=pop[k] * a[k], minlength=ncode)
        Ef = np.bincount(code, weights=pop[0] * a[k], minlength=ncode)
        PMP = np.bincount(code, weights=pmc * pop[k], minlength=ncode)
        D = np.bincount(code, weights=np.where(np.isfinite(deaths[k]), deaths[k], 0.0), minlength=ncode)
        LC = np.bincount(code, weights=np.nan_to_num(H_C[k]), minlength=ncode)
        LA = np.bincount(code, weights=np.nan_to_num(H_A[k]), minlength=ncode)
        for c in range(1, ncode):
            rows.append({"iso3": iso_of[c], "year": y, "pop": P[c], "pm25_popw": PMP[c] / P[c] if P[c] > 0 else np.nan,
                         "deaths": D[c], "loss_A": LA[c], "loss_C": LC[c]})
            frows.append({"iso3": iso_of[c], "year": y, "P": P[c], "P2000": P0[c], "E": E[c], "E_fix": Ef[c], "deaths": D[c]})
            for j, al in enumerate(ages):
                arows.append({"iso3": iso_of[c], "year": y, "age_lower": al, "deaths": dage[k, c, j]})
    pd.DataFrame(rows).to_csv(os.path.join(td, "panel_country_year.csv"), index=False)
    pd.DataFrame(frows).to_csv(os.path.join(td, "factors_country_year.csv"), index=False)
    pd.DataFrame(arows).to_csv(os.path.join(td, "deaths_country_age_year.csv"), index=False)
    fac = ai[["iso3", "year", "age_lower", "pshare"]].copy()
    fac["M"] = ai[list(CAUSES)].sum(axis=1)
    fac.to_csv(os.path.join(td, "age_inputs_factors.csv"), index=False)

    # ---- trends and transition states
    r_pm, ok_pm = tr.relative_trend(np.where(land, pm, np.nan))
    r_h, ok_h = tr.relative_trend(np.where(land, H_C, np.nan))
    r_d, _ = tr.relative_trend(np.where(land, deaths, np.nan))
    r_p, _ = tr.relative_trend(np.where(land, pop, np.nan))
    state = tr.classify_states(np.where(ok_pm, r_pm, np.nan), np.where(ok_h, r_h, np.nan))
    state[~land] = 0
    key = code * NS + state
    nk = ncode * NS
    k19, k00 = yi[2019], yi[2000]
    srows = []
    S = {"pop2019": np.bincount(key, weights=pop[k19], minlength=nk),
         "pop2000": np.bincount(key, weights=pop[k00], minlength=nk),
         "loss2019": np.bincount(key, weights=np.nan_to_num(H_C[k19]), minlength=nk),
         "loss2000": np.bincount(key, weights=np.nan_to_num(H_C[k00]), minlength=nk),
         "npx": np.bincount(key, minlength=nk).astype(float),
         "pop2019_deaths_declining": np.bincount(key, weights=np.where(tr.declining(r_d), pop[k19], 0.0), minlength=nk),
         "pop2000_deaths_declining": np.bincount(key, weights=np.where(tr.declining(r_d), pop[k00], 0.0), minlength=nk),
         "pop2019_pop_declining": np.bincount(key, weights=np.where(tr.declining(r_p), pop[k19], 0.0), minlength=nk)}
    for c in range(1, ncode):
        for s in range(NS):
            i = c * NS + s
            if S["npx"][i] > 0:
                srows.append({"iso3": iso_of[c], "state": s, **{n: float(v[i]) for n, v in S.items()}})
    pd.DataFrame(srows).to_csv(os.path.join(td, "states_country.csv"), index=False)

    rows = []
    for y, k in yi.items():
        pmc = np.where(np.isfinite(pm[k]), pm[k], 0.0)
        acc = {"P": pop[k], "P2000": pop[0], "E": pop[k] * a[k], "E_fix": pop[0] * a[k],
               "deaths": np.where(np.isfinite(deaths[k]), deaths[k], 0.0), "loss_C": np.nan_to_num(H_C[k]),
               "pm_x_pop": pmc * pop[k]}
        sums = {n: np.bincount(key, weights=v, minlength=nk) for n, v in acc.items()}
        for c in range(1, ncode):
            for s in range(1, NS):
                i = c * NS + s
                if S["npx"][i] > 0:
                    rows.append({"iso3": iso_of[c], "state": s, "year": y, **{n: float(v[i]) for n, v in sums.items()}})
    pd.DataFrame(rows).to_csv(os.path.join(td, "factors_state_year.csv"), index=False)

    # ---- fixed-exposure scenario
    pm_base = np.mean([pm[yi[y]] for y in BASE_YEARS], axis=0)
    a_base = np.where(land, af_fn(pm_base), 0.0)
    rows = []
    gap_sum = np.zeros(code.size)
    hcf19 = None
    for y in range(BASE_YEARS[-1], 2020):
        k = yi[y]
        h = np.nan_to_num(H_C[k])
        hcf, gap = sc.fixed_exposure(h, a[k], a_base)
        if y in POST_YEARS:
            gap_sum += gap
        if y == 2019:
            hcf19 = hcf
        H = np.bincount(code, weights=h, minlength=ncode)
        Hc = np.bincount(code, weights=hcf, minlength=ncode)
        for c in range(1, ncode):
            rows.append({"iso3": iso_of[c], "year": y, "H": H[c], "H_cf": Hc[c]})
    pd.DataFrame(rows).to_csv(os.path.join(td, "scenario_country_year.csv"), index=False)
    gap_post = gap_sum / len(POST_YEARS)
    h19 = np.nan_to_num(H_C[k19])
    acc = {"H2019": h19, "Hcf2019": hcf19, "gap_post": gap_post, "gap_pos_post": np.maximum(gap_post, 0.0),
           "pop2019": pop[k19]}
    sums = {n: np.bincount(key, weights=v, minlength=nk) for n, v in acc.items()}
    rows = [{"iso3": iso_of[c], "state": s, **{n: float(v[c * NS + s]) for n, v in sums.items()}}
            for c in range(1, ncode) for s in range(NS) if S["npx"][c * NS + s] > 0]
    pd.DataFrame(rows).to_csv(os.path.join(td, "scenario_state.csv"), index=False)

    # ---- required reductions and WHO levels
    pm19 = pm[k19]
    hk = sc.k_curve(h19, pm19, a[k19], af_fn)
    hc = sc.cap_curve(h19, pm19, a[k19], af_fn)
    rows_c, rows_s, rows_cap = [], [], []
    for j, kk in enumerate(sc.K_GRID):
        bc = np.bincount(code, weights=hk[j], minlength=ncode)
        bs = np.bincount(key, weights=hk[j], minlength=nk)
        rows_c += [{"iso3": iso_of[c], "k": float(kk), "H_k": bc[c]} for c in range(1, ncode)]
        rows_s += [{"iso3": iso_of[c], "state": s, "k": float(kk), "H_k": bs[c * NS + s]}
                   for c in range(1, ncode) for s in range(1, NS) if S["npx"][c * NS + s] > 0]
    pop_ok = np.isfinite(pm19)
    for j, cap in enumerate(sc.WHO_LEVELS):
        bc = np.bincount(code, weights=hc[j], minlength=ncode)
        above = np.bincount(code, weights=np.where(pop_ok & (pm19 > cap), pop[k19], 0.0), minlength=ncode)
        rows_cap += [{"iso3": iso_of[c], "cap": cap, "H_cap": bc[c], "pop_above_cap": above[c]} for c in range(1, ncode)]
    pd.DataFrame(rows_c).to_csv(os.path.join(td, "kcurve_country.csv"), index=False)
    pd.DataFrame(rows_s).to_csv(os.path.join(td, "kcurve_state.csv"), index=False)
    pd.DataFrame(rows_cap).to_csv(os.path.join(td, "caps_country.csv"), index=False)

    # ---- three-term grid-cell decomposition, 2000-2019
    env, dem, val, okp = dc.pixel_three_terms(np.nan_to_num(H_C[k00]), h19, np.nan_to_num(deaths[k00]),
                                              np.nan_to_num(deaths[k19]), a[k00], a[k19])
    split = np.where(tr.declining(r_d), 1, 2)          # 1 deaths declining, 2 not
    key2 = key * 3 + split
    rows = []
    for n, v in (("env", env), ("dem", dem), ("val", val), ("H0", np.where(okp, np.nan_to_num(H_C[k00]), 0.0)),
                 ("H1", np.where(okp, h19, 0.0))):
        b = np.bincount(key2, weights=v, minlength=nk * 3)
        for c in range(1, ncode):
            for s in range(NS):
                for dsp in (1, 2):
                    rows.append({"iso3": iso_of[c], "state": s, "deaths_trend": "declining" if dsp == 1 else "not_declining",
                                 "term": n, "value": float(b[(c * NS + s) * 3 + dsp])})
    pt = pd.DataFrame(rows).pivot_table(index=["iso3", "state", "deaths_trend"], columns="term", values="value").reset_index()
    pt.to_csv(os.path.join(td, "pixel_terms_state.csv"), index=False)
    largest = np.argmax(np.abs(np.vstack([env, dem, val])), axis=0)
    np.savez_compressed(os.path.join(args.out, "maps.npz"), shape=np.array(g["country"].shape), state=state,
                        pop2019=pop[k19], gap_post=gap_post, largest_term=np.where(okp, largest + 1, 0))
    print(f"step 1 done: tables in {td}")


if __name__ == "__main__":
    main()
