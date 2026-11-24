"""Normalized expected severity loss used by the study notebooks."""

import abc
import warnings

from ..bbn import bbn_autonomous_ranked_generated as bbn


class CostModel(abc.ABC):

    """Define the contract for scalar cost computation from a BBN.
    
    The retained concrete implementation is NormalizedExpectedLoss.
    """
    DEFAULT_EVENTS = [
        {
            "name": "grounding",
            "prob_accessor": "get_grounding",
            "sev_accessor": "get_grounding_severity",
        },
        {
            "name": "collision",
            "prob_accessor": "get_collision",
            "sev_accessor": "get_collision_severity",
        },
    ]

    SEVERITY_LABELS = ["negligible", "marginal", "moderate", "critical", "catastrophic"]

    @abc.abstractmethod
    def compute_cost(self, bbn_network) -> float:
        """Return a scalar cost from the network's current posterior distributions."""
        ...

    def compute_worst_case_cost(self, bbn_network: bbn.BayesianBeliefNetwork) -> float:
        """Evaluate cost after forcing catastrophic severity and likely accidents.
        
        Mutates evidence on the supplied network and updates beliefs before computing
        cost. Subsequent point queries clear and replace this evidence.
        """

        bbn_network.set_grounding_severity(bbn.Grounding_Severity.CATASTROPHIC)
        bbn_network.set_collision_severity(bbn.Collision_Severity.CATASTROPHIC)
        bbn_network.set_collision(bbn.Collision.LIKELY)
        bbn_network.set_grounding(bbn.Grounding.LIKELY)

        bbn_network.update_beliefs()

        return self.compute_cost(bbn_network)

class NormalizedExpectedLoss(CostModel):
    """Combine normalized expected severity losses across hazards.
    
    Parameters
    ----------
    weights : sequence, optional
        Non-decreasing severity weights with a positive maximum. Default values are
        ``[0, 4, 16, 64, 256]`` for negligible through catastrophic severity.
        A nonzero negligible weight produces a warning because it adds background
        cost throughout the map.
    hazards : dict, optional
        Mapping from hazard name to a severity accessor method on the network.
        Defaults to grounding and collision.
    
    Notes
    -----
    Severity marginals already contain accident probability; it is not multiplied
    in again. Each hazard's expected weight is divided by the largest weight.
    The combined risk is one minus the product of the hazards' residual risks.
    For valid distributions and nonnegative weights starting at zero it is bounded
    to [0, 1]. This is a severity-weighted index, not an accident probability.
    
    Examples
    --------
    Customize consequence weights without changing the network::
    
        loss = NormalizedExpectedLoss(
            weights=[0, 1, 4, 16, 64],
            hazards={"grounding": "get_grounding_severity",
                     "collision": "get_collision_severity"},
        )
    """

    DEFAULT_WEIGHTS = [0.0, 4.0, 16.0, 64.0, 256.0]

    # name -> severity_accessor. No event accessor: that is the whole point.
    DEFAULT_HAZARDS = {
        "grounding": "get_grounding_severity",
        "collision": "get_collision_severity",
    }

    def __init__(self, weights=None, hazards=None):
        self.weights = list(weights or self.DEFAULT_WEIGHTS)
        self.hazards = dict(hazards or self.DEFAULT_HAZARDS)

        if any(b < a for a, b in zip(self.weights, self.weights[1:])):
            raise ValueError(f"severity weights must be non-decreasing, got {self.weights}")

        self._wmax = max(self.weights)
        if self._wmax <= 0.0:
            raise ValueError("the most severe outcome must carry a positive weight")

        if self.weights[0] != 0.0:
            warnings.warn(
                "NormalizedExpectedLoss: a non-zero weight on 'negligible' adds a "
                "near-constant offset to every node, which the planner sees as extra "
                "path-length weight rather than risk.",
                stacklevel=2,
            )

    def _hazard_risk(self, sev_dist) -> float:
        """Expected severity weight for one hazard, normalized to [0, 1]."""
        return sum(p * w for p, w in zip(sev_dist, self.weights)) / self._wmax

    def compute_risk(self, bbn_network) -> dict:
        """Return combined loss and the loss contribution of each hazard.
        
        The result is ``{"combined": scalar, "per_hazard": {name: scalar}}``.
        """
        per_hazard = {
            name: self._hazard_risk(list(getattr(bbn_network, accessor)()))
            for name, accessor in self.hazards.items()
        }

        # Bounded probabilistic-OR: hazards compound, the result stays in [0,1],
        # and it degrades gracefully to max() when one hazard dominates.
        residual = 1.0
        for r in per_hazard.values():
            residual *= (1.0 - r)

        return {"combined": 1.0 - residual, "per_hazard": per_hazard}

    def compute_cost(self, bbn_network) -> float:
        """Single [0,1] scalar for the planner."""
        return self.compute_risk(bbn_network)["combined"]
