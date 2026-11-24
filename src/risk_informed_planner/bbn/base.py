import abc
import enum
from collections.abc import Mapping, Sequence

import pysmile


class BaseBayesianBeliefNetworkNode(abc.ABC):
    """Abstract contract for BBN node helpers used by generated APIs."""

    node_id: str
    states: enum.Enum | None = None

    @abc.abstractmethod
    def set_evidence(self, state: str | int) -> None:
        """Set node evidence using state id or state index."""

    @abc.abstractmethod
    def get_value(self) -> Sequence[float]:
        """Return posterior probabilities for this node."""


class BaseBayesianBeliefNetwork(abc.ABC):
    """Abstract contract for pySMILE-backed Bayesian network wrappers."""

    network: pysmile.Network

    @abc.abstractmethod
    def clear_all_evidence(self) -> None:
        """Clear all previously set evidence in the network."""

    @abc.abstractmethod
    def update_beliefs(self) -> None:
        """Trigger pySMILE belief propagation."""

    @abc.abstractmethod
    def set_evidence(self, node_id: str, state: str | int) -> None:
        """Set evidence for a node using node id and state id or index."""

    @abc.abstractmethod
    def set_virtual_evidence(
        self, node_id: str, distribution: Mapping[str | int, float] | Sequence[float]
    ) -> None:
        """Set virtual (soft) evidence for a node as a likelihood over its states."""

    @abc.abstractmethod
    def get_virtual_evidence(self, node_id: str) -> list[float]:
        """Return the virtual evidence currently set on a node."""

    @abc.abstractmethod
    def is_virtual_evidence(self, node_id: str) -> bool:
        """Return True if the node currently holds virtual evidence."""

    @abc.abstractmethod
    def clear_evidence(self, node_id: str) -> None:
        """Clear evidence (hard or virtual) for a single node."""

    @abc.abstractmethod
    def get_state_index(self, node_id: str, state_id: str) -> int:
        """Resolve a state id to a pySMILE outcome index."""

    @abc.abstractmethod
    def get_node_value(self, node_id: str) -> list[float]:
        """Return posterior probabilities for a node id."""

    @abc.abstractmethod
    def inference(self, node_id: str) -> list[float]:
        """Run inference and return posterior values for a target node."""

    @abc.abstractmethod
    def snapshot(self) -> dict[str, dict]:
        """Return a snapshot of all nodes with their values and state names."""