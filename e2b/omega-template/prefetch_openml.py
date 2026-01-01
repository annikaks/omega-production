import openml

openml.config.cache_directory = "/data/openml_cache"
data_ids = [
    1463,  # Balance Scale
    1464,  # Blood Transfusion
    43,    # Haberman
    1499,  # Seeds
    48,    # Teaching Assistant
    62,    # Zoo
    1490,  # Planning Relax
    59,    # Ionosphere
    40,    # Sonar
    41,    # Glass
    54,    # Vehicle
    1459,  # Liver Disorders
    53,    # Heart Statlog
    37,    # Pima Indians Diabetes
    40945, # Australian
    333,   # Monks-1
]
for data_id in data_ids:
    try:
        openml.datasets.get_dataset(data_id)
    except Exception as exc:
        print(f"prefetch failed for {data_id}: {exc}")
print("openml prefetch done")
