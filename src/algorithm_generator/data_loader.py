# src/algorithm_generator/data_loader.py

from __future__ import annotations

from typing import Dict, Tuple, List, Optional
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_digits,
    fetch_openml,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def get_openml_classification_ids() -> Dict[str, Tuple[int, Optional[int]]]:
    return {
        "Adult": (1590, 50000),        # large, cap rows
        "Bank Marketing": (1461, None),
        "Credit-G": (31, None),
        "Phoneme": (1489, None),
        "Spambase": (44, None),
        "Ionosphere": (59, None),
        "Sonar": (40, None),
        "Vehicle": (54, None),
        "Glass": (41, None),
    }



def load_openml_classification_dataset(
    data_id: int,
    test_size: float = 0.2,
    random_state: int = 42,
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = fetch_openml(data_id=data_id, as_frame=True, return_X_y=True)

    if max_rows is not None and len(X) > max_rows:
        X = X.sample(n=max_rows, random_state=random_state)
        y = y.loc[X.index]

    y = y.astype("category")
    y_enc = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=test_size,
        random_state=random_state,
        stratify=y_enc,
    )

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in numeric_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )


    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # force dense
    )


    X_train_out = pre.fit_transform(X_train)
    X_test_out = pre.transform(X_test)

    # ensure dense
    if hasattr(X_train_out, "toarray"):
        X_train_out = X_train_out.toarray()
    if hasattr(X_test_out, "toarray"):
        X_test_out = X_test_out.toarray()

    # ensure numeric dtype for all models
    X_train_out = np.asarray(X_train_out, dtype=np.float32)
    X_test_out = np.asarray(X_test_out, dtype=np.float32)


    return X_train_out, X_test_out, y_train, y_test


def load_sklearn_builtin_classification_dataset(
    name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    builtins = {
        "Iris": load_iris(return_X_y=True),
        "Wine": load_wine(return_X_y=True),
        "Breast Cancer": load_breast_cancer(return_X_y=True),
        "Digits": load_digits(return_X_y=True),
    }

    if name not in builtins:
        raise ValueError(f"Unknown sklearn builtin dataset: {name}")

    X, y = builtins[name]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def load_classification_datasets(
    dataset_names: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    logging: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:

    openml_ids = get_openml_classification_ids()
    split_datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for name in dataset_names:
        # sklearn built-ins
        if name in {"Iris", "Wine", "Breast Cancer", "Digits"}:
            split_datasets[name] = load_sklearn_builtin_classification_dataset(
                name=name,
                test_size=test_size,
                random_state=random_state,
            )
            if logging:
                print(f"Loaded builtin dataset: {name}")
            continue

        # OpenML datasets (with caps)
        if name in openml_ids:
            data_id, max_rows = openml_ids[name]
            split_datasets[name] = load_openml_classification_dataset(
                data_id=data_id,
                test_size=test_size,
                random_state=random_state,
                max_rows=max_rows,
            )
            if logging:
                print(f"Loaded OpenML dataset: {name} (cap={max_rows})")
            continue

        raise ValueError(
            f"Dataset '{name}' not supported (builtin + OpenML only)."
        )

    return split_datasets

