# thread caps
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import inspect
import importlib.util
import warnings
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
from scipy.sparse import SparseEfficiencyWarning

from data_loader import load_classification_datasets


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.manifold._isomap")


@dataclass(frozen=True)
class BenchmarkTask:
    model: Any
    model_name: str
    dataset_name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def eval_one_benchmark_task(
    task: BenchmarkTask,
) -> Tuple[str, str, Dict[str, float], Optional[str], Dict[str, float]]:
    """
    Returns:
      (model_name, dataset_name, cell_dict, error_msg_or_None, stats_dict)

    cell_dict:
      {"Accuracy": float} on success
      {"error": "..."} on failure
    """
    t0 = time.perf_counter()
    fit_s: Optional[float] = None
    pred_s: Optional[float] = None

    try:
        fresh_model = clone(task.model)

        X_train = np.asarray(task.X_train)
        X_test = np.asarray(task.X_test)

        t_fit0 = time.perf_counter()
        fresh_model.fit(X_train, task.y_train)
        fit_s = time.perf_counter() - t_fit0

        t_pred0 = time.perf_counter()
        y_pred = fresh_model.predict(X_test)
        pred_s = time.perf_counter() - t_pred0

        score = accuracy_score(task.y_test, y_pred)

        total_s = time.perf_counter() - t0
        stats = {
            "total_s": float(total_s),
            "fit_s": float(fit_s),
            "pred_s": float(pred_s),
            "timed_out": 0.0, 
        }
        return (task.model_name, task.dataset_name, {"Accuracy": float(score)}, None, stats)

    except Exception as e:
        total_s = time.perf_counter() - t0
        msg = f"{task.model_name} failed on {task.dataset_name}: {type(e).__name__}: {e}"
        stats = {
            "total_s": float(total_s),
            "fit_s": float(fit_s) if fit_s is not None else -1.0,
            "pred_s": float(pred_s) if pred_s is not None else -1.0,
            "timed_out": 0.0,
        }
        return (task.model_name, task.dataset_name, {"error": msg}, msg, stats)


class BenchmarkSuite:
    def __init__(
        self,
        dataset_names: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        logging: bool = False,
        debugging: bool = False,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.dataset_names = dataset_names or ["Iris", "Wine", "Breast Cancer", "Digits"]

        self.datasets = load_classification_datasets(
            dataset_names=self.dataset_names,
            test_size=self.test_size,
            random_state=self.random_state,
            logging=logging,
        )

        self.logging = logging
        self.debugging = debugging

        self.results: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.runtime_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    def run_benchmark(self, models: List[BaseEstimator], n_jobs: int = 4) -> Dict[str, Dict[str, Dict[str, float]]]:
        # initialize results so downstream printing works
        self.results = {m.__class__.__name__: {} for m in models}
        self.runtime_stats = {}

        tasks: List[BenchmarkTask] = []
        for model in models:
            model_name = model.__class__.__name__
            for dataset_name, (X_train, X_test, y_train, y_test) in self.datasets.items():
                tasks.append(
                    BenchmarkTask(
                        model=model,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                    )
                )

        outputs = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(eval_one_benchmark_task)(t) for t in tasks
        )

        failed = 0
        for model_name, dataset_name, cell, err, stats in outputs:
            self.results.setdefault(model_name, {})
            self.results[model_name][dataset_name] = cell

            self.runtime_stats.setdefault(model_name, {})
            self.runtime_stats[model_name][dataset_name] = stats

            if err is not None:
                failed += 1
                if self.logging and self.debugging:
                    print(err)

        if self.logging:
            total = len(tasks)
            print(f"Benchmark done: {total - failed}/{total} succeeded, {failed} failed")

        return self.results

    def compute_aggregate_relative_score_strict(self):
        datasets = list(self.datasets.keys())
        models = list(self.results.keys())

        raw_by_dataset: Dict[str, Dict[str, float]] = {d: {} for d in datasets}
        for d in datasets:
            for m in models:
                cell = self.results.get(m, {}).get(d, {})
                if "Accuracy" in cell:
                    raw_by_dataset[d][m] = float(cell["Accuracy"])

        norm_by_dataset: Dict[str, Dict[str, float]] = {d: {} for d in datasets}
        for d in datasets:
            vals = list(raw_by_dataset[d].values())
            if not vals:
                continue
            mn, mx = min(vals), max(vals)
            denom = mx - mn
            for m, s in raw_by_dataset[d].items():
                norm_by_dataset[d][m] = 1.0 if denom == 0 else (s - mn) / denom

        aggregate: Dict[str, float] = {}
        for m in models:
            total = 0.0
            for d in datasets:
                total += norm_by_dataset[d].get(m, 0.0)
            aggregate[m] = total / len(datasets) if datasets else 0.0

        return aggregate, norm_by_dataset

    def print_table(self):
        datasets = list(self.datasets.keys())
        models = list(self.results.keys())

        aggregate, _ = self.compute_aggregate_relative_score_strict()

        colw = max(12, max(len(d) for d in datasets) + 2)
        roww = max(24, max(len(m) for m in models) + 2)

        header = (
            "Model".ljust(roww)
            + "".join(d.ljust(colw) for d in datasets)
            + "Aggregate".ljust(colw)
        )
        print(header)
        print("-" * len(header))

        models_sorted = sorted(models, key=lambda m: aggregate.get(m, 0.0), reverse=True)

        for model_name in models_sorted:
            row = model_name.ljust(roww)
            for dataset_name in datasets:
                cell = self.results.get(model_name, {}).get(dataset_name, {})
                if "Accuracy" in cell:
                    row += f"{cell['Accuracy']:.4f}".ljust(colw)
                else:
                    row += "ERR".ljust(colw)
            row += f"{aggregate.get(model_name, 0.0):.3f}".ljust(colw)
            print(row)

    def summarize_runtime(self):
        summary: Dict[str, Dict[str, float]] = {}
        for model, by_ds in self.runtime_stats.items():
            times = [v["total_s"] for v in by_ds.values() if v.get("total_s", -1) >= 0]
            summary[model] = {
                "mean_total_s": float(np.mean(times)) if times else float("inf"),
                "max_total_s": float(np.max(times)) if times else float("inf"),
                "timeouts": 0,  # no enforced timeouts in this version
            }
        return summary

    def save_latex_table_multirow(
        self,
        filepath: str,
        caption: str = "Benchmark results",
        label: str = "tab:benchmark",
    ):
        datasets = list(self.datasets.keys())
        models = list(self.results.keys())

        aggregate, _ = self.compute_aggregate_relative_score_strict()
        models_sorted = sorted(models, key=lambda mn: aggregate.get(mn, 0.0), reverse=True)

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        with open(filepath, "w") as f:
            f.write("% Auto-generated by BenchmarkSuite.save_latex_table_multirow (classification-only)\n")
            f.write("% Requires LaTeX packages: booktabs, multirow\n\n")

            f.write("\\begin{table}[ht]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{{caption}}}\n")
            f.write(f"\\label{{{label}}}\n")

            f.write("\\begin{tabular}{l" + "c" * (len(datasets) + 1) + "}\n")
            f.write("\\toprule\n")

            f.write(
                "Model"
                + f" & \\multicolumn{{{len(datasets)}}}{{c}}{{Classification}}"
                + " & \\multicolumn{1}{c}{RelAgg}"
                + " \\\\\n"
            )
            f.write(f"\\cmidrule(lr){{2-{1 + len(datasets)}}}\n")

            header2 = " "
            for d in datasets:
                header2 += f" & \\multicolumn{{1}}{{c}}{{{d}}}"
            header2 += " & "
            f.write(header2 + " \\\\\n")

            header3 = " "
            for _ in datasets:
                header3 += " & Accuracy"
            header3 += " & Rel"
            f.write(header3 + " \\\\\n")

            f.write("\\midrule\n")

            for model_name in models_sorted:
                row = model_name
                for d in datasets:
                    cell = self.results.get(model_name, {}).get(d, {})
                    if "Accuracy" in cell:
                        row += f" & {cell['Accuracy']:.4f}"
                    else:
                        row += " & --"
                row += f" & {aggregate.get(model_name, 0.0):.3f}"
                f.write(row + " \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"Saved LaTeX table to: {filepath}")
        print(r"!!! Make sure to include \usepackage{booktabs} and \usepackage{multirow}")

    def save_runtime_stats_csv(self, filepath: str):
        import csv
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        with open(filepath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "dataset", "total_s", "fit_s", "pred_s", "timed_out"])

            for model_name, by_ds in self.runtime_stats.items():
                for dataset_name, stats in by_ds.items():
                    w.writerow([
                        model_name,
                        dataset_name,
                        stats.get("total_s", ""),
                        stats.get("fit_s", ""),
                        stats.get("pred_s", ""),
                        stats.get("timed_out", ""),
                    ])


def load_models_from_directory(models_dir: str, logging: bool = False):
    models: List[BaseEstimator] = []

    for filename in sorted(os.listdir(models_dir)):
        if not filename.endswith(".py"):
            continue
        if filename == "__init__.py":
            continue

        file_path = os.path.join(models_dir, filename)
        module_name = os.path.splitext(filename)[0]

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if not hasattr(obj, "fit") or not hasattr(obj, "predict"):
                continue

            try:
                instance = obj()
                models.append(instance)
                if logging:
                    print(f"Loaded model: {obj.__name__} from {filename}")
            except Exception as e:
                if logging:
                    print(f"Skipped {obj.__name__} (could not instantiate): {e}")

    return models


def main():
    logging = False

    current_dir = os.path.dirname(os.path.abspath(__file__))
    metaomni_dir = os.path.join(current_dir, "metaomni")
    models_dirs = [metaomni_dir]

    suite = BenchmarkSuite(
        dataset_names=[
            "Iris",
            "Breast Cancer",
            "Credit-G",
            "Phoneme",
            "Vehicle",
            "Glass",
        ],
        logging=logging,
        debugging=False,
    )

    models: List[BaseEstimator] = []
    for d in models_dirs:
        py_count = len([f for f in os.listdir(d) if f.endswith(".py")])
        print(f"loading ~{py_count} .py files from {d}")
        models.extend(load_models_from_directory(d, logging=logging))

    if not models:
        print("No valid models found.")
        return

    suite.run_benchmark(models, n_jobs=4)

    logs_dir = os.path.join(metaomni_dir, "evaluation_logs")
    os.makedirs(logs_dir, exist_ok=True)

    suite.save_runtime_stats_csv(os.path.join(logs_dir, "runtime_task_stats.csv"))

    runtime_summary = suite.summarize_runtime()
    pd.DataFrame.from_dict(runtime_summary, orient="index").to_csv(
        os.path.join(logs_dir, "runtime_summary.csv")
    )

    suite.print_table()

    suite.save_latex_table_multirow(
        filepath=os.path.join(logs_dir, "benchmark_multirow.tex"),
        caption="MetaOmni-generated models evaluated on classification datasets.",
        label="tab:metaomni-classification-benchmark",
    )


if __name__ == "__main__":
    main()
