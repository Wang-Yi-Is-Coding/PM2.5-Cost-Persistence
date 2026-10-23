"""pmcp: methods for comparing PM2.5 exposure with monetized mortality loss.

The modules implement the equations in the Methods of the accompanying paper:

    attributable   cause-weighted attributable fractions and age allocation
    valuation      base and age-adjusted VSL and monetized loss
    trends         Theil-Sen trends, transition states and aggregate series
    decomposition  seven-factor LMDI-I decomposition and the grid-cell three-term version
    scenarios      fixed-exposure scenario and required exposure reductions
    tables         aggregation from grid cells to country, state and group tables

All functions operate on NumPy arrays or pandas tables and do not read raster files.
"""

__version__ = "1.0.0"

CAUSES = ("COPD", "LC", "LRI", "IHD", "STR")
YEARS = tuple(range(2000, 2020))
STATE_NAMES = {
    0: "Not analysed",
    1: "Cost persistence",
    2: "Co-decline",
    3: "Loss-only decline",
    4: "Dual rise",
}
