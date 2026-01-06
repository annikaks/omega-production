import importlib.util
import numpy as np

dataset_names = [
    "Iris",
    "Wine",
    "Breast Cancer",
    "Digits",
    "Balance Scale",
    "Blood Transfusion",
    "Haberman",
    "Seeds",
    "Teaching Assistant",
    "Zoo",
    "Planning Relax",
    "Ionosphere",
    "Sonar",
    "Glass",
    "Vehicle",
    "Liver Disorders",
    "Heart Statlog",
    "Pima Indians Diabetes",
    "Australian",
    "Monks-1",
]

spec = importlib.util.spec_from_file_location("data_loader", "/opt/template/data_loader.py")
data_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_loader)

datasets = data_loader.load_classification_datasets(dataset_names)
for name, (X_train, X_test, y_train, y_test) in datasets.items():
    safe = name.replace(" ", "_").replace("/", "_")
    out_path = f"/data/benchmarks/{safe}.npz"
    np.savez(
        out_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
print("benchmark cache written")
