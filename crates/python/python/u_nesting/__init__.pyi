"""Type stubs for u_nesting module."""

from typing import Any, Dict, List, Literal, Optional, TypedDict

class Geometry2D(TypedDict, total=False):
    """2D geometry input."""
    id: str
    polygon: List[List[float]]
    holes: Optional[List[List[List[float]]]]
    quantity: int
    rotations: Optional[List[float]]
    allow_flip: bool

class Boundary2D(TypedDict, total=False):
    """2D boundary input."""
    width: Optional[float]
    height: Optional[float]
    polygon: Optional[List[List[float]]]

class Geometry3D(TypedDict, total=False):
    """3D geometry input."""
    id: str
    dimensions: List[float]
    quantity: int
    mass: Optional[float]
    orientation: Optional[str]

class Boundary3D(TypedDict, total=False):
    """3D boundary input."""
    dimensions: List[float]
    max_mass: Optional[float]
    gravity: bool
    stability: bool

class Config(TypedDict, total=False):
    """Solver configuration."""
    strategy: Optional[str]
    spacing: Optional[float]
    margin: Optional[float]
    time_limit_ms: Optional[int]
    target_utilization: Optional[float]
    population_size: Optional[int]
    max_generations: Optional[int]
    max_iterations: Optional[int]
    crossover_rate: Optional[float]
    mutation_rate: Optional[float]
    seed: Optional[int]
    multi_sheet: Optional[bool]

class Placement(TypedDict):
    """Placement result."""
    geometry_id: str
    instance: int
    position: List[float]
    rotation: List[float]
    boundary_index: int

class SolveResult(TypedDict):
    """Solve operation result. Every refusal is reported here, never raised."""
    success: bool
    placements: List[Placement]
    boundaries_used: int
    utilization: float
    total_requested: int
    unplaced: List[str]
    unplaced_count: int
    all_placed: bool
    used_bounding_box: List[float]
    used_utilization: float
    stop_reason: Optional[
        Literal[
            "completed",
            "optimal",
            "target_reached",
            "iteration_limit",
            "no_improvement",
            "time_limit",
            "cancelled",
        ]
    ]
    """Why the run stopped; ``None`` on a refusal. ``time_limit`` means the layout
    is the best found when ``time_limit_ms`` ran out."""
    computation_time_ms: int
    error: Optional[str]
    code: Optional[str]
    """Stable reason a solve was refused (``malformed_input``, ``value_not_finite``,
    ``parameter_out_of_range``, ``invalid_geometry``, ``duplicate_id``, ...)."""
    details: Optional[Dict[str, Any]]
    """The values behind ``code``: ``parameter``, ``id``, ``index``, ``min``, ``max``, ``got``, ..."""

def solve_2d(
    geometries: List[Geometry2D],
    boundary: Boundary2D,
    config: Optional[Config] = None,
) -> SolveResult:
    """
    Solve a 2D nesting problem.

    Args:
        geometries: List of geometry dictionaries.
        boundary: Boundary specification.
        config: Optional solver configuration.

    Returns:
        Dictionary containing solve results.
    """
    ...

def solve_3d(
    geometries: List[Geometry3D],
    boundary: Boundary3D,
    config: Optional[Config] = None,
) -> SolveResult:
    """
    Solve a 3D bin packing problem.

    Args:
        geometries: List of geometry dictionaries.
        boundary: Boundary specification.
        config: Optional solver configuration.

    Returns:
        Dictionary containing solve results.
    """
    ...

def version() -> str:
    """Get the library version."""
    ...

class StrategyInfo(TypedDict):
    """One strategy of ``available_strategies()``."""
    name: str
    reads: List[str]

StrategyCatalog = TypedDict("StrategyCatalog", {"2d": List[StrategyInfo], "3d": List[StrategyInfo]})

def available_strategies() -> StrategyCatalog:
    """The strategies each dimension offers and the settings each reads.

    ``{"2d": [{"name": "blf", "reads": ["spacing", ...]}, ...], "3d": [...]}``
    -- the table a solve's refusal of a setting the strategy does not read
    (``invalid_option``) comes from, so a caller can offer only the settings
    that change the chosen strategy's run.
    """
    ...
