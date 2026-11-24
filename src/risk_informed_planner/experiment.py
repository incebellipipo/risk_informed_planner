"""Saved planner runs for reproducing figures without rerunning inference."""
import dataclasses
import pickle
from pathlib import Path

from .utils import Node
from .config import Config



class _PlannerRunUnpickler(pickle.Unpickler):
    """Read saved runs created before the package rename."""

    def find_class(self, module, name):
        legacy_package = "risk_based_planner"
        if module == legacy_package or module.startswith(legacy_package + "."):
            module = "risk_informed_planner" + module[len(legacy_package):]
        if module == "risk_informed_planner.risk_exploration_angle_cost":
            module = "risk_informed_planner.planner"
        return super().find_class(module, name)


@dataclasses.dataclass
class PlannerRun:
    """Store a planner run, its configuration, and the data needed for figures.
    
    Parameters
    ----------
    nodes : list
        Explored nodes, including disconnected risk observations.
    path : list
        Ordered root-to-goal route; an empty list denotes no retained route.
    cfg : Config
        Configuration used by the planner.
    graph : object
        Loaded cartoblobpy graph, or None when no graph is attached.
    obstacles : list, optional
        Static and moving obstacle definitions. Each run has its own default list.
    t_computation : float, optional
        Planning duration in seconds.
    speed : float, optional
        Constant own-vessel speed in metres per second.
    map_file : str, optional
        Map YAML path, resolved to an absolute path during initialization.
    metadata : dict, optional
        Experiment annotations, with an independent default dictionary.
    
    Notes
    -----
    The risk model and native BBN are not stored. Node ``bbn_snapshot`` dictionaries
    hold the recorded posteriors. With ``map_file``, saving omits the graph and
    retains its start/goal coordinates; loading reconstructs it from the same
    absolute map path. Without ``map_file``, the attached graph is pickled in full.
    
    The loader remaps the former ``risk_based_planner`` package name. Other class
    attributes, dependencies, and map paths must remain compatible. These pickles
    support local plotting iterations rather than long-term archival. Only load
    trusted files: pickle deserialization can execute code.
    
    Examples
    --------
    Save a completed experiment, then redraw it without native inference::
    
        run = PlannerRun(nodes, path, cfg, graph, obstacles=obstacles,
                         t_computation=elapsed, speed=3.0, map_file=map_file)
        run.save("results/study_03.pkl")
        restored = PlannerRun.load("results/study_03.pkl")
        panels = plot_paper_panels(**restored.plot_kwargs(), modes=("heatmap",))
    """
    nodes: list[Node]
    path: list[Node]
    cfg: Config
    graph: object  # cartoblobpy.graph.Graph
    obstacles: list = dataclasses.field(default_factory=list)
    t_computation: float = None
    speed: float = None
    map_file: str = None
    metadata: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        # absolute, so the run loads from any working directory
        if self.map_file is not None:
            self.map_file = str(Path(self.map_file).resolve())

    def __getstate__(self):
        state = dict(self.__dict__)
        if self.map_file is not None and state["graph"] is not None:
            graph = state.pop("graph")
            state["graph"] = None
            state["_graph_start_goal"] = (graph.start, graph.goal)
        return state

    def __setstate__(self, state):
        start_goal = state.pop("_graph_start_goal", None)
        self.__dict__.update(state)
        if start_goal is not None:
            from cartoblobpy.graph import Graph

            self.graph = Graph()
            self.graph.load_from_yaml(self.map_file)
            self.graph.start, self.graph.goal = start_goal

    def save(self, file_path) -> Path:
        """Save this run and return the destination Path.
        
        Parent folders are created automatically. Pickle preserves shared node
        identity and parent links. The original graph remains attached to this object.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return file_path

    @classmethod
    def load(cls, file_path) -> "PlannerRun":
        """Load one run, remapping legacy module paths and reloading its map if needed.
        
        Raises
        ------
        TypeError
            If the file does not contain a PlannerRun.
        
        Notes
        -----
        Only load trusted pickle files. A saved map reference must still be available
        at its absolute path; missing map inputs propagate the map-loader error.
        """
        with open(file_path, "rb") as f:
            run = _PlannerRunUnpickler(f).load()
        if not isinstance(run, cls):
            raise TypeError(f"{file_path} holds a {type(run).__name__}, not a {cls.__name__}")
        return run

    @staticmethod
    def save_many(runs: list["PlannerRun"], file_path) -> Path:
        """Save a campaign list and return the destination Path.
        
        Create parent folders and preserve run order and shared objects within the
        campaign. An empty campaign is supported.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(list(runs), f, protocol=pickle.HIGHEST_PROTOCOL)
        return file_path

    @classmethod
    def load_many(cls, file_path) -> list["PlannerRun"]:
        """Load a campaign, preserving run order and shared object identity.
        
        Raises
        ------
        TypeError
            If the payload is not a list containing only PlannerRun objects.
        
        Notes
        -----
        Supports legacy module names and map references as in ``load``. Only load
        trusted pickle files.
        """
        with open(file_path, "rb") as f:
            runs = _PlannerRunUnpickler(f).load()
        if not isinstance(runs, list) or not all(isinstance(run, cls) for run in runs):
            raise TypeError(f"{file_path} does not hold a list of {cls.__name__}")
        return runs

    def plot_kwargs(self) -> dict:
        """Return plotting keyword arguments shared by matrix and paper-panel helpers.
        
        The mapping contains ``nodes``, ``path``, ``cfg``, ``graph``, ``obstacles``,
        ``t_computation``, and ``speed``.
        """
        return dict(
            nodes=self.nodes, path=self.path, cfg=self.cfg, graph=self.graph,
            obstacles=self.obstacles, t_computation=self.t_computation, speed=self.speed,
        )