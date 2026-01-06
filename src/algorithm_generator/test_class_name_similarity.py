import unittest

import api
import sandbox_queue


class ClassNameSimilarityTests(unittest.TestCase):
    def test_find_similar_names(self):
        candidates = ["RandomForestClassifier", "LogisticRegressionClassifier", "MetaSynthesisClassifier"]
        similar = api._find_similar_names("RandomForestClassifier", candidates, limit=5, threshold=0.8)
        self.assertIn("RandomForestClassifier", similar)

    def test_find_similar_names_sandbox_queue(self):
        manager = sandbox_queue.SandboxQueueManager.__new__(sandbox_queue.SandboxQueueManager)
        candidates = {"VotingEnsembleClassifier", "StackedEnsembleClassifier"}
        similar = manager._find_similar_names("VotingEnsembleClassifier", candidates, limit=5, threshold=0.8)
        self.assertIn("VotingEnsembleClassifier", similar)


if __name__ == "__main__":
    unittest.main()
