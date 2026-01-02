import json
from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
BOUNDS_PATH = STORAGE_DIR / "bounds.json"
STORAGE_DIR.mkdir(exist_ok=True)


def read_json(path: Path, default: Any):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def update_bounds_and_calculate_score(new_metrics: Dict[str, float]) -> float:
    bounds = read_json(BOUNDS_PATH, {})

    clean_metrics = {
        k: (
            float(v)
            if isinstance(v, (int, float, str)) and str(v).replace(".", "", 1).isdigit()
            else 0.0
        )
        for k, v in new_metrics.items()
    }

    bounds_changed = False
    for ds, val in clean_metrics.items():
        if ds not in bounds:
            bounds[ds] = {"min": val, "max": val}
            bounds_changed = True
        else:
            if val < bounds[ds]["min"]:
                bounds[ds]["min"] = val
                bounds_changed = True
            if val > bounds[ds]["max"]:
                bounds[ds]["max"] = val
                bounds_changed = True

    if bounds_changed:
        write_json_atomic(BOUNDS_PATH, bounds)

    rel_scores = []
    for ds, val in clean_metrics.items():
        mn, mx = bounds[ds]["min"], bounds[ds]["max"]
        denom = mx - mn
        score = (val - mn) / denom if denom > 0 else 1.0
        rel_scores.append(score)

    return sum(rel_scores) / len(rel_scores) if rel_scores else 0.0
