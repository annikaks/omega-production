import argparse
from typing import List

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from evaluate import BenchmarkSuite


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test for evaluation and scoring."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["Iris", "Wine", "Breast Cancer", "Digits"],
        help="Dataset names to evaluate (defaults to sklearn built-ins).",
    )
    return parser.parse_args()


def run_evaluation(dataset_names: List[str]) -> None:
    suite = BenchmarkSuite(dataset_names=dataset_names, logging=True, debugging=True)
    models = [
        DummyClassifier(strategy="most_frequent"),
        LogisticRegression(max_iter=500, n_jobs=1),
    ]
    suite.run_benchmark(models=models, n_jobs=4)
    aggregate, _ = suite.compute_aggregate_relative_score_strict()

    print("Aggregate scores:")
    for model_name, score in aggregate.items():
        print(f"{model_name}: {score:.4f}")


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(args.datasets)
