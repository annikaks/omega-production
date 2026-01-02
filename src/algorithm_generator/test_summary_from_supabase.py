import asyncio
import unittest

import api


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    def __init__(self, store, name):
        self.store = store
        self.name = name
        self._select = None
        self._filters = []
        self._single = False
        self._update = None

    def select(self, columns):
        self._select = [c.strip() for c in columns.split(",")] if columns else None
        return self

    def eq(self, key, value):
        self._filters.append((key, value))
        return self

    def single(self):
        self._single = True
        return self

    def update(self, payload):
        self._update = payload
        return self

    def execute(self):
        rows = list(self.store.get(self.name, []))
        for key, value in self._filters:
            rows = [row for row in rows if row.get(key) == value]
        if self._update is not None:
            for row in rows:
                row.update(self._update)
            return _FakeResponse(rows[0] if self._single and rows else rows)
        if self._select:
            selected = []
            for row in rows:
                selected.append({col: row.get(col) for col in self._select})
            rows = selected
        return _FakeResponse(rows[0] if self._single and rows else rows)


class _FakeSupabase:
    def __init__(self, algorithms):
        self.store = {"algorithms": algorithms}

    def table(self, name):
        return _FakeTable(self.store, name)


class _FakeAnalyzer:
    def __init__(self, summaries):
        self.summaries = summaries

    def describe_code(self, code):
        return self.summaries[code]


class SummarizeSupabaseTests(unittest.TestCase):
    def test_summary_saved_from_algorithm_code(self):
        meta_code = "class MetaSynthesisClassifier:\n    pass\n"
        voting_code = "class VotingEnsembleClassifier:\n    pass\n"
        algorithms = [
            {"id": "meta-id", "summary": None, "file_name": "meta.py", "algorithm_code": meta_code},
            {"id": "voting-id", "summary": None, "file_name": "voting.py", "algorithm_code": voting_code},
        ]
        summaries = {
            meta_code: "MetaSynthesisClassifier : summary text",
            voting_code: "VotingEnsembleClassifier : summary text",
        }

        api.supabase = _FakeSupabase(algorithms)
        api.analyzer = _FakeAnalyzer(summaries)

        result = asyncio.run(api.get_summary("meta-id"))
        self.assertEqual(result["summary"], summaries[meta_code])
        self.assertEqual(algorithms[0]["summary"], summaries[meta_code])

        result = asyncio.run(api.get_summary("voting-id"))
        self.assertEqual(result["summary"], summaries[voting_code])
        self.assertEqual(algorithms[1]["summary"], summaries[voting_code])


if __name__ == "__main__":
    unittest.main()
