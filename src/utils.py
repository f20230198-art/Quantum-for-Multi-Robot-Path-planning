"""
utils.py  –  Shared Utilities
==============================

Common helper functions used across multiple planners.
"""

import math
from typing import List, Tuple

Position = Tuple[int, int]


def path_cost(path: List[Position]) -> float:
    """Euclidean cost of a path (diagonal moves cost sqrt(2), cardinal moves cost 1.0)."""
    cost = 0.0
    for i in range(len(path) - 1):
        dr = abs(path[i][0] - path[i + 1][0])
        dc = abs(path[i][1] - path[i + 1][1])
        cost += math.sqrt(2) if (dr + dc == 2) else 1.0
    return cost
