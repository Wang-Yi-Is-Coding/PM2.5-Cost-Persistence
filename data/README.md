# Data

## Example data (`data/example/`)

Written by `scripts/make_example_data.py`. All values are synthetic.

| File | Columns / arrays |
|---|---|
| `countries.csv` | iso3, name, income_group (World Bank labels, e.g. `High income`), region |
| `gdp_pc.csv` | iso3, year, gdp_pc (PPP GDP per capita, current international $; gaps allowed), oecd_gdp_pc (same-year OECD aggregate) |
| `age_inputs.csv` | iso3, year, age_lower, pshare (population share of the age group), ex (remaining life expectancy at the start of the group), COPD, LC, LRI, IHD, STR (death rates per person-year) |
| `wealth_weights.csv` | age (25, 30, …, 85), omega |
| `af_table.csv` | pm25 (µg m⁻³), AF_COPD, AF_LC, AF_LRI, AF_IHD, AF_STR (attributable fractions, 0–1) |
| `grid.npz` | years; country (grid of country codes, 0 = not analysed); country_iso3 (code → iso3); pm25, pop, deaths (years × rows × columns) |

## Study inputs

The paper used the following sources (see the Methods of the paper):

| Input | Source |
|---|---|
| Annual PM2.5, 0.01° | NASA SEDAC, Global Annual PM2.5 Grids V4.GL.03 (https://sedac.ciesin.columbia.edu) |
| Attributable deaths, 1 km, central and bound estimates | Published mortality product described in Xia et al. (2026), *Commun. Earth Environ.* 7, 754 (https://doi.org/10.1038/s43247-026-03722-6) |
| Population, 1 km | WorldPop (https://hub.worldpop.org) |
| Age-specific population, life tables | GBD 2019 (https://vizhub.healthdata.org/gbd-results) |
| Age- and cause-specific death rates | GBD 2023 (https://vizhub.healthdata.org/gbd-results) |
| PPP GDP per capita, OECD aggregate | World Bank World Development Indicators, NY.GDP.PCAP.PP.CD (https://databank.worldbank.org) |
| Income groups and regions | World Bank country classifications |
| Wealth weights by age | Yin et al. (2021), *Lancet Planetary Health* 5, e356–e367, published data files |
| Attributable-fraction lookup | Integrated exposure–response functions of Burnett et al. (2014), *Environ. Health Perspect.* 122, 397–403 |

For the study, the life-table groups <1 and 1–4 years were merged into one <5 group by population-weighted averaging after the age-adjusted VSL was computed (`valuation.collapse_age_groups`). The reason is that the GBD 2023 cause-specific death rates are reported for <5 years.

## Study outputs

The tables written by `scripts/step1_grid_to_tables.py` for the study have the following names (where they are available is given in the paper's Data availability statement):
- `panel_country_year.csv`
- `factors_country_year.csv`
- `deaths_country_age_year.csv`
- `age_inputs_factors.csv`
- `valuation_country_year_age.csv`
- `states_country.csv`
- `factors_state_year.csv`
- `scenario_country_year.csv`
- `scenario_state.csv`
- `kcurve_country.csv`
- `kcurve_state.csv`
- `caps_country.csv`
- `pixel_terms_state.csv`

Monetary values are in 2011 PPP US dollars.
