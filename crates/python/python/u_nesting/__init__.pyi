"""Type stubs for u_nesting module."""

from typing import Any, Dict, List, Optional, TypedDict

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
    crossover_rate: Optional[float]
    mutation_rate: Optional[float]

class Placement(TypedDict):
    """Placement result."""
    geometry_id: str
    instance: int
    position: List[float]
    rotation: List[float]
    boundary_index: int

class SolveResult(TypedDict):
    """Solve operation result."""
    success: bool
    placements: List[Placement]
    boundaries_used: int
    utilization: float
    unplaced: List[str]
    computation_time_ms: int
    error: Optional[str]
    code: Optional[str]
    """Stable reason a solve was refused (``invalid_geometry``, ``duplicate_id``, ...)."""
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

def available_strategies() -> List[str]:
    """List available optimization strategies.

    Returns the union of 2D and 3D strategy names: ``nfp``, ``gdrr`` and
    ``alns`` apply to 2D nesting, ``ep`` to 3D packing, the rest to both.
    """
    ...
