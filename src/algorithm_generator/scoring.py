from typing import Dict, Iterable, List, Tuple


DATASET_COLUMNS: List[Tuple[str, str]] = [
    ("Iris", "iris_acc"),
    ("Wine", "wine_acc"),
    ("Breast Cancer", "breast_cancer_acc"),
    ("Digits", "digits_acc"),
    ("Balance Scale", "balance_scale_acc"),
    ("Blood Transfusion", "blood_transfusion_acc"),
    ("Haberman", "haberman_acc"),
    ("Seeds", "seeds_acc"),
    ("Teaching Assistant", "teaching_assistant_acc"),
    ("Zoo", "zoo_acc"),
    ("Planning Relax", "planning_relax_acc"),
    ("Ionosphere", "ionosphere_acc"),
    ("Sonar", "sonar_acc"),
    ("Glass", "glass_acc"),
    ("Vehicle", "vehicle_acc"),
    ("Liver Disorders", "liver_disorders_acc"),
    ("Heart Statlog", "heart_statlog_acc"),
    ("Pima Indians Diabetes", "pima_diabetes_acc"),
    ("Australian", "australian_acc"),
    ("Monks-1", "monks_1_acc"),
]


def _coerce_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def compute_bounds_from_rows(rows: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    bounds: Dict[str, Dict[str, float]] = {}
    for dataset, column in DATASET_COLUMNS:
        values: List[float] = []
        for row in rows:
            if column in row and row[column] is not None:
                values.append(_coerce_float(row[column]))
        if values:
            bounds[dataset] = {"min": min(values), "max": max(values)}
        else:
            bounds[dataset] = {"min": 0.0, "max": 0.0}
    return bounds


def calculate_min_max_score(metrics: Dict[str, float], bounds: Dict[str, Dict[str, float]]) -> float:
    rel_scores: List[float] = []
    for dataset, _column in DATASET_COLUMNS:
        val = _coerce_float(metrics.get(dataset))
        mn = bounds.get(dataset, {}).get("min", 0.0)
        mx = bounds.get(dataset, {}).get("max", 0.0)
        denom = mx - mn
        score = (val - mn) / denom if denom > 0 else 1.0
        rel_scores.append(score)
    return sum(rel_scores) / len(rel_scores) if rel_scores else 0.0


def fetch_bounds_from_supabase(supabase) -> Dict[str, Dict[str, float]]:
    columns = ",".join([column for _dataset, column in DATASET_COLUMNS])
    res = supabase.table("algorithms").select(columns).execute()
    rows = res.data or []
    return compute_bounds_from_rows(rows)


def recompute_min_max_scores(supabase) -> Dict[str, Dict[str, float]]:
    columns = ",".join(["id"] + [column for _dataset, column in DATASET_COLUMNS])
    res = supabase.table("algorithms").select(columns).execute()
    rows = res.data or []
    bounds = compute_bounds_from_rows(rows)
    for row in rows:
        metrics = {
            dataset: _coerce_float(row.get(column))
            for dataset, column in DATASET_COLUMNS
        }
        score = calculate_min_max_score(metrics, bounds)
        supabase.table("algorithms").update({"min_max_score": score}).eq("id", row["id"]).execute()
    return bounds


def recompute_min_max_scores_for_table(
    supabase,
    table_name: str,
) -> Dict[str, Dict[str, float]]:
    columns = ",".join(["id", "algorithm_code"] + [column for _dataset, column in DATASET_COLUMNS])
    res = supabase.table(table_name).select(columns).execute()
    rows = res.data or []
    bounds = compute_bounds_from_rows(rows)
    for row in rows:
        algorithm_code = row.get("algorithm_code") or ""
        if isinstance(algorithm_code, str) and algorithm_code.startswith("ERROR:"):
            continue
        metrics = {
            dataset: _coerce_float(row.get(column))
            for dataset, column in DATASET_COLUMNS
        }
        score = calculate_min_max_score(metrics, bounds)
        supabase.table(table_name).update({"min_max_score": score}).eq("id", row["id"]).execute()
    return bounds
