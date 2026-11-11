"""
U-Nesting: 2D/3D Spatial Optimization Engine

A high-performance library for 2D polygon nesting and 3D bin packing.

Example:
    >>> import u_nesting
    >>>
    >>> # 2D Nesting
    >>> result = u_nesting.solve_2d(
    ...     geometries=[
    ...         {"id": "rect", "polygon": [[0,0], [100,0], [100,50], [0,50]], "quantity": 5}
    ...     ],
    ...     boundary={"width": 500, "height": 300},
    ...     config={"strategy": "nfp"}
    ... )
    >>> print(f"Utilization: {result['utilization']:.1%}")
    >>> # Judge the layout by the rules the nester places under
    >>> u_nesting.check_2d(
    ...     geometries=[
    ...         {"id": "rect", "polygon": [[0,0], [100,0], [100,50], [0,50]], "quantity": 5}
    ...     ],
    ...     boundary={"width": 500, "height": 300},
    ...     solve_result=result,
    ... )["feasible"]
    True

    >>> # 3D Bin Packing
    >>> result = u_nesting.solve_3d(
    ...     geometries=[
    ...         {"id": "box", "dimensions": [100, 50, 30], "quantity": 10}
    ...     ],
    ...     boundary={"dimensions": [500, 400, 300]},
    ...     config={"strategy": "ep"}
    ... )
"""

from .u_nesting import (
    solve_2d,
    solve_3d,
    check_2d,
    check_3d,
    optimize_cutting_path,
    version,
    available_strategies,
    codes,
)

__all__ = [
    "solve_2d",
    "solve_3d",
    "check_2d",
    "check_3d",
    "optimize_cutting_path",
    "version",
    "available_strategies",
    "codes",
]
__version__ = version()
