import unittest

from scoring import DATASET_COLUMNS, calculate_min_max_score, recompute_min_max_scores


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    def __init__(self, store, name):
        self.store = store
        self.name = name
        self._columns = None
        self._filters = []
        self._update_payload = None

    def select(self, columns, **_kwargs):
        self._columns = [c.strip() for c in columns.split(",")] if columns else None
        return self

    def update(self, payload):
        self._update_payload = payload
        return self

    def eq(self, key, value):
        self._filters.append((key, value))
        return self

    def execute(self):
        rows = list(self.store.get(self.name, []))
        for key, value in self._filters:
            rows = [row for row in rows if row.get(key) == value]
        if self._update_payload is not None:
            for row in rows:
                row.update(self._update_payload)
            return _FakeResponse(rows)
        if self._columns:
            selected = []
            for row in rows:
                selected.append({col: row.get(col) for col in self._columns})
            return _FakeResponse(selected)
        return _FakeResponse(rows)


class _FakeSupabase:
    def __init__(self, algorithms):
        self.store = {"algorithms": algorithms}

    def table(self, name):
        return _FakeTable(self.store, name)


class SupabaseBoundsTests(unittest.TestCase):
    def test_recompute_updates_all_min_max_scores(self):
        rows = [
            {"id": 1, "iris_acc": 0.5, "wine_acc": 0.7},
            {"id": 2, "iris_acc": 0.9, "wine_acc": 0.2},
        ]
        supabase = _FakeSupabase(rows)

        bounds = recompute_min_max_scores(supabase)
        self.assertIn("Iris", bounds)
        self.assertIn("Wine", bounds)

        for row in rows:
            metrics = {
                dataset: row.get(column, 0.0) or 0.0
                for dataset, column in DATASET_COLUMNS
            }
            expected = calculate_min_max_score(metrics, bounds)
            self.assertAlmostEqual(row["min_max_score"], expected, places=6)


if __name__ == "__main__":
    unittest.main()
