"""Write the synthetic example dataset to data/example/.

The example has three fictitious countries (AAA, BBB, CCC) on a 60 x 90 grid for 2000-2019. All values
are invented with closed-form expressions so that every step of the workflow can be run and checked
quickly. They are not estimates for any real place and must not be interpreted as results.

Usage:  python scripts/make_example_data.py [--out data/example]
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
YEARS = np.arange(2000, 2020)
T = len(YEARS)
NY, NX = 60, 90
CAUSES = ("COPD", "LC", "LRI", "IHD", "STR")
AGE_LOWER = [0] + list(range(5, 100, 5))          # 20 groups: <5, 5-9, ..., 95+

COUNTRIES = pd.DataFrame({
    "iso3": ["AAA", "BBB", "CCC"],
    "name": ["Example country A", "Example country B", "Example country C"],
    "income_group": ["High income", "Upper middle income", "Lower middle income"],
    "region": ["Europe & Central Asia", "East Asia & Pacific", "South Asia"],
})


def af_table():
    pm = np.round(np.arange(1.0, 500.01, 0.5), 2)
    z = np.maximum(pm - 2.4, 0.0)
    par = {"COPD": (0.8, 0.020, 0.90), "LC": (0.9, 0.015, 0.95), "LRI": (1.5, 0.030, 0.80),
           "IHD": (1.1, 0.050, 0.70), "STR": (1.0, 0.060, 0.65)}
    t = pd.DataFrame({"pm25": pm})
    for c, (alpha, gamma, delta) in par.items():
        rr = 1.0 + alpha * (1.0 - np.exp(-gamma * z ** delta))
        t[f"AF_{c}"] = 1.0 - 1.0 / rr
    return t


def gdp_table():
    rows = []
    level = {"AAA": (30000.0, 0.035), "BBB": (4000.0, 0.105), "CCC": (1800.0, 0.075)}
    for i3, (g0, g) in level.items():
        for k, y in enumerate(YEARS):
            v = g0 * (1 + g) ** k
            if i3 == "CCC" and y == 2006:
                v = np.nan                      # one missing year, filled by interpolation
            rows.append({"iso3": i3, "year": int(y), "gdp_pc": v})
    oecd = pd.DataFrame({"year": YEARS, "oecd_gdp_pc": 27000.0 * 1.035 ** np.arange(T)})
    return pd.DataFrame(rows).merge(oecd, on="year")


def age_inputs():
    """Age-specific population, remaining life expectancy and five cause-specific death rates."""
    rows = []
    prof = {"AAA": (0.020, 0.0006, 76.0, 0.18, 0.012), "BBB": (0.034, 0.0011, 70.0, 0.25, 0.022),
            "CCC": (0.040, 0.0004, 62.0, 0.20, 0.010)}
    base = {"COPD": 4e-4, "LC": 2e-4, "LRI": 1.5e-4, "IHD": 8e-4, "STR": 6e-4}
    for i3, (lam0, dlam, e0, gain, decline) in prof.items():
        for k, y in enumerate(YEARS):
            lam = lam0 - dlam * k                       # population ageing: flatter age profile
            a = np.array(AGE_LOWER, float)
            w = np.exp(-lam * a) * np.where(a >= 95, 0.4, 1.0)
            pshare = w / w.sum()
            e = e0 + gain * k
            ex = np.maximum(2.5, (e + 4.0) - 0.88 * a)
            for j, al in enumerate(AGE_LOWER):
                r = {"iso3": i3, "year": int(y), "age_lower": al, "pshare": pshare[j], "ex": ex[j]}
                for c in CAUSES:
                    m = base[c] * np.exp(0.075 * (al - 60)) * (1 - decline) ** k
                    if c == "LRI" and al == 0:
                        m += 3e-4 * (1 - 0.04) ** k
                    r[c] = m
                rows.append(r)
    return pd.DataFrame(rows)


def grid(af):
    yy, xx = np.mgrid[0:NY, 0:NX]
    country = np.where(xx < 30, 1, np.where(xx < 60, 2, 3)).astype(np.int16)
    country[0:6, 40:46] = 0                             # a lake: not analysed
    s = np.sin(yy / 7.0) * np.cos(xx / 9.0)
    dens = np.exp(1.8 * np.exp(-((yy - 30) ** 2 + (xx % 30 - 15) ** 2) / 120.0) + 0.6 * s)
    pop0 = np.choose(country, [0.0, 900.0, 1500.0, 1300.0]) * dens
    pm0 = np.choose(country, [np.nan, 18.0, 45.0, 60.0]) * (1 + 0.35 * s) * (1 + 0.004 * yy)
    t = np.arange(T)[:, None, None]
    peak = 8 + (yy / NY * 6).astype(int)                # B: peak year varies by row (2008-2013)
    pm = np.empty((T, NY, NX))
    pm[:] = pm0
    a_mask = country == 1
    b_mask = country == 2
    c_mask = country == 3
    pm = np.where(a_mask, pm0 * (1 - 0.03) ** t, pm)
    up = (1 + 0.035) ** np.minimum(t, peak)
    down = (1 - 0.075) ** np.maximum(t - peak, 0)
    pm = np.where(b_mask, pm0 * up * down, pm)
    trend_c = np.where(yy < 15, -0.008, 0.022)
    pm = np.where(c_mask, pm0 * (1 + trend_c) ** t, pm)
    pm = pm * (1 + 0.03 * np.sin(1.7 * t + xx / 5.0))     # year-to-year variation
    growth = np.where(a_mask, 0.003, np.where(b_mask, 0.004 + 0.02 * (dens > 2.5) - 0.012 * (dens < 1.0), 0.018))
    pop = pop0 * (1 + growth) ** t
    pm = np.where(country > 0, pm, np.nan).astype(np.float32)
    pop = np.where(country > 0, pop, 0.0).astype(np.float32)

    # "Published" attributable deaths: pop x sum_a p_a sum_k AF_k(PM) m_ak, with a small departure
    # from the national rates so that the residual calibration term is not trivially one.
    ai = age_inputs()
    pm_grid = af.pm25.to_numpy()
    afk = {c: af[f"AF_{c}"].to_numpy() for c in CAUSES}
    deaths = np.zeros((T, NY, NX), np.float32)
    for code, i3 in enumerate(["AAA", "BBB", "CCC"], start=1):
        m = country == code
        for k, y in enumerate(YEARS):
            a = ai[(ai.iso3 == i3) & (ai.year == y)]
            x = pm[k][m].astype(float)
            per_cap = np.zeros(x.size)
            for c in CAUSES:
                per_cap += np.interp(x, pm_grid, afk[c], left=0.0, right=afk[c][-1]) * float((a.pshare * a[c]).sum())
            wiggle = 1 + 0.05 * np.sin(0.9 * k + 0.3 * code + yy[m] / 11.0)
            deaths[k][m] = (pop[k][m] * per_cap * wiggle).astype(np.float32)
    deaths = np.where(country > 0, deaths, np.nan).astype(np.float32)
    return country, pm, pop, deaths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "example"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    af = af_table()
    af.to_csv(os.path.join(args.out, "af_table.csv"), index=False)
    COUNTRIES.to_csv(os.path.join(args.out, "countries.csv"), index=False)
    gdp_table().to_csv(os.path.join(args.out, "gdp_pc.csv"), index=False)
    age_inputs().to_csv(os.path.join(args.out, "age_inputs.csv"), index=False)
    ages = np.arange(25, 90, 5)
    omega = [0.30, 0.45, 0.65, 0.85, 1.05, 1.25, 1.40, 1.45, 1.40, 1.25, 1.05, 0.85, 0.70]
    pd.DataFrame({"age": ages, "omega": omega}).to_csv(os.path.join(args.out, "wealth_weights.csv"), index=False)
    country, pm, pop, deaths = grid(af)
    np.savez_compressed(os.path.join(args.out, "grid.npz"), years=YEARS, country=country,
                        country_iso3=np.array(["", "AAA", "BBB", "CCC"]), pm25=pm, pop=pop, deaths=deaths)
    print(f"example data written to {args.out}")


if __name__ == "__main__":
    main()
