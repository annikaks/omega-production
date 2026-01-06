from e2b import Template

OPENML_IDS = [
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

template = (
    Template()
    .from_image("e2bdev/base")
    .run_cmd(
        "python -m pip install --no-cache-dir numpy scikit-learn scipy pandas openml"
    )
    .run_cmd("mkdir -p /data/openml_cache")
    .run_cmd(
        "python - <<'PY'\n"
        "import openml\n"
        "openml.config.cache_directory = '/data/openml_cache'\n"
        "for data_id in "
        + str(OPENML_IDS)
        + ":\n"
        "    try:\n"
        "        openml.datasets.get_dataset(data_id)\n"
        "    except Exception as exc:\n"
        "        print(f'prefetch failed for {data_id}: {exc}')\n"
        "print('openml prefetch done')\n"
        "PY"
    )
)
