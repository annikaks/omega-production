import json
from pathlib import Path

# Paths
STORAGE_DIR = Path("storage")
BOUNDS_PATH = STORAGE_DIR / "bounds.json"
STORAGE_DIR.mkdir(exist_ok=True)

# Your Scikit-Learn Baseline Data
from storage.display_benchmarks import SKLEARN_SCORES

def initialize_bounds():
    bounds = {}
    datasets = SKLEARN_SCORES[0].keys()
    
    for ds in datasets:
        all_vals = [m[ds] for m in SKLEARN_SCORES]
        bounds[ds] = {
            "min": min(all_vals),
            "max": max(all_vals)
        }
    
    with open(BOUNDS_PATH, "w") as f:
        json.dump(bounds, f, indent=2)
    print(f"Successfully initialized boundaries for {len(datasets)} datasets.")

if __name__ == "__main__":
    initialize_bounds()