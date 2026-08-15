"""
DBSW R260003 Topology Engine - Core Package Initialization
Continuum domain discretization, Eurocode material registry, and 3D SIMP solver modules.

Horizon-2 Service Differentiator | Authorship Engineering Platform
"""

__version__ = "1.0.0"
__author__ = "Damian Brenlla / DBSW"

from .domain import Domain3D
from .materials import MATERIALS, get_material
from .solvers import run_simp_optimisation, hex8_stiffness_matrix

__all__ = [
    "Domain3D",
    "MATERIALS",
    "get_material",
    "run_simp_optimisation",
    "hex8_stiffness_matrix",
]
