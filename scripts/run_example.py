"""Run the full workflow on the synthetic example data (about one minute)."""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
STEPS = ["make_example_data.py", "step1_grid_to_tables.py", "step2_results.py", "make_figures.py"]

if __name__ == "__main__":
    for s in STEPS:
        print(f"== {s}", flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", s)], check=True, cwd=ROOT)
