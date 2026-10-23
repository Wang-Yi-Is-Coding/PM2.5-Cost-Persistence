# PM2.5-Cost-Persistence

Code accompanying the paper by Wang et al. on PM2.5 exposure and monetized mortality loss.

The paper compares 2000–2019 trends in exposure to fine particulate matter (PM2.5) with the monetized loss assigned to PM2.5-attributable deaths on a 1-km grid. It identifies **cost persistence**, where exposure declined but monetized loss did not. It decomposes the change in loss into seven factors and evaluates exposure scenarios.

This repository contains:
- the methods of the paper as a small Python package (`pmcp`);
- a workflow that runs from grid-cell arrays to the tables and figure panels;
- a synthetic example dataset, so that the workflow can be run end to end in about a minute.

## Contents

| Path | Purpose |
|---|---|
| `pmcp/attributable.py` | Attributable-fraction lookup, cause weights, cause-weighted attributable fraction a_c(PM), age allocation of attributable deaths |
| `pmcp/valuation.py` | Base VSL, age-adjusted VSL, monetized loss; gap filling of GDP series; wealth-weight mapping |
| `pmcp/trends.py` | Relative Theil–Sen slopes, the four transition states, peak years, Sen (1968) confidence intervals for aggregate series |
| `pmcp/decomposition.py` | Seven-factor identity and additive LMDI-I by country or country-by-state unit; grid-cell three-term decomposition |
| `pmcp/scenarios.py` | Fixed-exposure scenario, proportional-reduction curves, required reductions, WHO levels |
| `pmcp/tables.py` | Aggregation helpers, weighted median, Spearman correlation with Fisher-z interval |
| `scripts/make_example_data.py` | Writes the synthetic example dataset to `data/example/` |
| `scripts/step1_grid_to_tables.py` | Grid-cell arrays → valuation, panel, factor, state and scenario tables |
| `scripts/step2_results.py` | Tables → aggregate series, state shares, decomposition, scenario results (`summary.json`) |
| `scripts/make_figures.py` | Draft versions of the main figure panels |
| `scripts/run_example.py` | Runs the four scripts above in order |
| `tests/test_methods.py` | Checks against known values and identities (for example, LMDI closure) |
| `data/README.md` | Input formats, data sources and the formats of the study outputs |

## Quick start

```bash
python -m pip install -r requirements.txt
python tests/test_methods.py        # or: python -m pytest tests
python scripts/run_example.py       # writes outputs/example/{tables,results,figures}
```

The example uses three fictitious countries on a 60 × 90 grid. Its values are invented; they demonstrate the workflow and are not results.

## Equations and code

| Method | Function |
|---|---|
| Age allocation, D_{i,a} = D_i × s_a(PM_i) | `attributable.age_shares` |
| Base VSL | `valuation.base_vsl`, `valuation.income_elasticity` |
| Age-adjusted VSL, model C | `valuation.age_adjusted_vsl`, `valuation.wealth_weight_age` |
| Monetized loss | `valuation.monetize` |
| Trends and transition states | `trends.relative_trend`, `trends.classify_states` |
| Seven-factor identity and LMDI-I | `decomposition.factor_panel`, `decomposition.lmdi` |
| Grid-cell three-term decomposition | `decomposition.pixel_three_terms` |
| Fixed-exposure scenario | `scenarios.fixed_exposure` |
| Required reductions | `scenarios.k_curve`, `scenarios.required_reductions` |
| Aggregate-series slopes with Sen (1968) intervals | `trends.sen_slope_ci` |

## Using the study data

`step2_results.py` reads only the tables written by step 1 (formats in `data/README.md`). The country, state and scenario tables produced for the paper use these formats; where they are available is given in the paper's Data availability statement. To run step 2 on them, place the tables in a `tables/` subfolder and point `--out` at its parent folder.

## Scope

The repository covers the main analyses: valuation, trend classification, decomposition, the fixed-exposure scenario and the required reductions. The source datasets are publicly available (see `data/README.md`).

## Requirements

Python 3.12 with NumPy, pandas, SciPy and Matplotlib (versions in `requirements.txt`). The analyses in the paper were run with NumPy 2.5.3, pandas 2.3.3, SciPy 1.18.1 and Matplotlib 3.11.2. No step uses random numbers.

## Citation and licence

Please cite the accompanying paper and this code (`CITATION.cff`). The code is released under the MIT licence (`LICENSE`). Input datasets remain under their own licences.

## Contact

Jizhe Xia (xiajizhe@szu.edu.cn)
