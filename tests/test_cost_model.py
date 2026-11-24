"""Loss behavior against analytical distributions and probability invariants."""
import unittest

import numpy as np

from tests.helpers import severity_network
from risk_informed_planner.risk.cost_model import NormalizedExpectedLoss


class ExpectedLossTests(unittest.TestCase):
    def test_benign_and_catastrophic_limits(self):
        for probabilities, expected in [([1, 0, 0, 0, 0], 0), ([0, 0, 0, 0, 1], 1)]:
            with self.subTest(probabilities=probabilities):
                self.assertEqual(NormalizedExpectedLoss().compute_cost(
                    severity_network(probabilities, [1, 0, 0, 0, 0])), expected)

    def test_marginal_severity_is_not_multiplied_by_accident_probability_again(self):
        network = severity_network([0.75, 0, 0, 0, 0.25], [1, 0, 0, 0, 0])
        self.assertAlmostEqual(NormalizedExpectedLoss().compute_cost(network), 0.25)

    def test_hazards_combine_without_exceeding_one(self):
        result = NormalizedExpectedLoss().compute_risk(
            severity_network([0.75, 0, 0, 0, 0.25], [0.5, 0, 0, 0, 0.5]))
        self.assertEqual(result["per_hazard"], {"grounding": 0.25, "collision": 0.5})
        self.assertAlmostEqual(result["combined"], 0.625)

    def test_custom_weights_and_hazards(self):
        model = NormalizedExpectedLoss(weights=[0, 1, 2, 3, 4],
                                       hazards={"only_grounding": "get_grounding_severity"})
        result = model.compute_risk(severity_network([0, 0.5, 0.5, 0, 0], None))
        self.assertEqual(result["per_hazard"], {"only_grounding": 0.375})
        self.assertAlmostEqual(result["combined"], 0.375)

    def test_weight_scale_does_not_change_normalized_loss(self):
        network = severity_network([0.1, 0.2, 0.3, 0.1, 0.3], [0.5, 0.2, 0.1, 0.1, 0.1])
        a = NormalizedExpectedLoss(weights=[0, 1, 2, 3, 4])
        b = NormalizedExpectedLoss(weights=[0, 10, 20, 30, 40])
        self.assertAlmostEqual(a.compute_cost(network), b.compute_cost(network))

    def test_loss_increases_with_severity(self):
        losses = []
        for severity in np.eye(5):
            losses.append(NormalizedExpectedLoss().compute_cost(
                severity_network(severity, [1, 0, 0, 0, 0])))
        self.assertTrue(all(a < b for a, b in zip(losses, losses[1:])))

    def test_random_probability_distributions_remain_bounded(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            result = NormalizedExpectedLoss().compute_risk(
                severity_network(rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))))
            self.assertTrue(all(0 <= r <= 1 for r in result["per_hazard"].values()))
            self.assertGreaterEqual(result["combined"], max(result["per_hazard"].values()))
            self.assertLessEqual(result["combined"], 1)

    def test_invalid_weight_curves_are_rejected(self):
        for weights in [[0, 4, 3, 2, 1], [0, 0, 0, 0, 0]]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                NormalizedExpectedLoss(weights=weights)

    def test_nonzero_benign_weight_warns(self):
        with self.assertWarnsRegex(UserWarning, "non-zero weight"):
            NormalizedExpectedLoss(weights=[1, 2, 3, 4, 5])
