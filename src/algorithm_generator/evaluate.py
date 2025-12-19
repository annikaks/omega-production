from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
)

from sklearn.base import BaseEstimator, clone
from typing import List, Tuple, Dict
import importlib.util
import inspect
import os

# (classification only)
class BenchmarkSuite:
    def __init__(self, dataset_names=None, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.dataset_names = dataset_names or ["Iris", "Wine", "Breast Cancer", "Digits"]
        self.datasets = self._load_datasets()
        self.results: Dict[str, Dict[str, Dict[str, float]]] = {}

    def _load_datasets(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        # classification-only datasets
        datasets = {
            "Iris": load_iris(return_X_y=True),
            "Wine": load_wine(return_X_y=True),
            "Breast Cancer": load_breast_cancer(return_X_y=True),
            "Digits": load_digits(return_X_y=True),
        }

        split_datasets = {}
        for name in self.dataset_names:
            if name not in datasets:
                raise ValueError(
                    f"Dataset '{name}' not supported. Classification-only supported: {list(datasets.keys())}"
                )

            X, y = datasets[name]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            split_datasets[name] = (X_train_scaled, X_test_scaled, y_train, y_test)
        
        return split_datasets

    def run_benchmark(self, models: List[BaseEstimator], logging:bool = False) -> Dict[str, Dict[str, float]]:
        for model in models:
            model_name = model.__class__.__name__
            self.results[model_name] = {}

            for dataset_name, (X_train, X_test, y_train, y_test) in self.datasets.items():
                try:
                    fresh_model = clone(model)
                    fresh_model.fit(X_train, y_train)
                    y_pred = fresh_model.predict(X_test)
                    score = accuracy_score(y_test, y_pred)
                    self.results[model_name][dataset_name] = {"Accuracy": score}
                except Exception as e:
                    msg = f"{model_name} failed on {dataset_name}: {type(e).__name__}: {e}"
                    if logging:
                        print(msg)
                    self.results[model_name][dataset_name] = {"error": msg}

        return self.results

    def compute_aggregate_relative_score_strict(self):
        """
        !! If I randomly pick one of these datasets, how close is this model to the best-performing model on that dataset?
        n_{m,d} = (s_{m,d} - min_d) / (max_d - min_d)  # s_{m,d} = models accuracy on dataset d
        RelAgg_m = (1 / |D|) * Σ_d n_{m,d}
        
        Strict aggregate score:
        - normalize model scores into [0,1] using min-max over successful models (/dataset)
        - any ERR/missing scores = 0.0 for that dataset
        - Aggregate is the mean normalized score across ALL datasets (equal weight)

        Returns:
        aggregate: dict[model_name] -> float
        norm_by_dataset: dict[dataset_name] -> dict[model_name] -> float
        """
        datasets = list(self.datasets.keys())
        models = list(self.results.keys())

        raw_by_dataset = {d: {} for d in datasets}
        for d in datasets:
            for m in models:
                cell = self.results.get(m, {}).get(d, {})
                if "Accuracy" in cell:
                    raw_by_dataset[d][m] = float(cell["Accuracy"])

        norm_by_dataset = {d: {} for d in datasets}
        for d in datasets:
            vals = list(raw_by_dataset[d].values())
            if not vals:
                continue
            mn, mx = min(vals), max(vals)
            denom = mx - mn
            for m, s in raw_by_dataset[d].items():
                norm_by_dataset[d][m] = 1.0 if denom == 0 else (s - mn) / denom

        aggregate = {}
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
        models = sorted(models, key=lambda m: aggregate.get(m, 0.0), reverse=True)

        for model_name in models:
            row = model_name.ljust(roww)
            for dataset_name in datasets:
                cell = self.results.get(model_name, {}).get(dataset_name, {})
                if "Accuracy" in cell:
                    row += f"{cell['Accuracy']:.4f}".ljust(colw)
                else:
                    row += "ERR".ljust(colw)
            row += f"{aggregate.get(model_name, 0.0):.3f}".ljust(colw)
            print(row)



    def _get_metric_label(self, dataset_name: str):
        for model_results in self.results.values():
            cell = model_results.get(dataset_name, {})
            if "Accuracy" in cell:
                return "Accuracy"
            if "R2" in cell:
                return "R²"
        return "Metric"
    
    

    def _get_metric_label(self, dataset_name: str):
        for model_results in self.results.values():
            cell = model_results.get(dataset_name, {})
            if "Accuracy" in cell:
                return "Accuracy"
            if "R2" in cell:
                return "R²"
        return "Metric"
    
    def save_latex_table_multirow( self, filepath: str, caption: str = "Benchmark results", label: str = "tab:benchmark",):
        datasets = list(self.datasets.keys())
        models = list(self.results.keys())

        aggregate, _ = self.compute_aggregate_relative_score_strict()
        models = sorted(models, key=lambda mn: aggregate.get(mn, 0.0), reverse=True)

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        with open(filepath, "w") as f:
            f.write("% Auto-generated by BenchmarkSuite.save_latex_table_multirow (classification-only)\n")
            f.write("% Requires LaTeX packages: booktabs, multirow\n\n")

            f.write("\\begin{table}[ht]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{{caption}}}\n")
            f.write(f"\\label{{{label}}}\n")

            # aggregate col
            f.write("\\begin{tabular}{l" + "c" * (len(datasets) + 1) + "}\n")
            f.write("\\toprule\n")

            # row 1: group
            f.write(
                "Model"
                + f" & \\multicolumn{{{len(datasets)}}}{{c}}{{Classification}}"
                + " & \\multicolumn{1}{c}{RelAgg}"
                + " \\\\\n"
            )
            f.write(f"\\cmidrule(lr){{2-{1 + len(datasets)}}}\n")

            # row 2: dataset names
            header2 = " "
            for d in datasets:
                header2 += f" & \\multicolumn{{1}}{{c}}{{{d}}}"
            header2 += " & "
            f.write(header2 + " \\\\\n")

            # row 3: metric labels
            header3 = " "
            for _ in datasets:
                header3 += " & Accuracy"
            header3 += " & Rel"
            f.write(header3 + " \\\\\n")

            f.write("\\midrule\n")

            for model_name in models:
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
        print("!!!Make sure to include \\\\usepackage{booktabs} and \\\\usepackage{multirow}")



def load_models_from_directory(models_dir: str, logging: bool=False):
    """
    Dynamically loads sklearn-style models from all .py files in a directory.
    Returns: list of model instances
    """
    models = []

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

            try: # try instantiating
                instance = obj()
                models.append(instance)
                if logging: print(f"Loaded model: {obj.__name__} from {filename}")
            except Exception as e:
                if logging: print(f"Skipped {obj.__name__} (could not instantiate): {e}")

    return models


def main():
    logging = False
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    metaomni_dir = os.path.join(current_dir, "metaomni")
    models_dir = [metaomni_dir]

    suite = BenchmarkSuite(
        dataset_names=["Iris", "Wine", "Breast Cancer", "Digits"],
    )

    models = []
    for dir in models_dir:
        print(f"loading {len(os.listdir(dir))} files from {dir}") 
        models.extend(load_models_from_directory(dir, logging))
    

    if not models:
        print("No valid models found.")
        return

    suite.run_benchmark(models, logging)
    suite.print_table()

    suite.save_latex_table_multirow(
        filepath="src/algorithm_generator/results/benchmark_multirow_25_12_18.tex",
        caption="MetaOmni-generated models evaluated on classification datasets.",
        label="tab:metaomni-classification-benchmark",
    )

if __name__ == "__main__":
    main()
