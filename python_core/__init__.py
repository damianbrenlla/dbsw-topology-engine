# DBSW Core Package Initialisation
# Author: Damian Brenlla / DBSW 2026

from core.domain import Domain3D
from core.materials import EurocodeMaterialRegistry
from core.solvers import TopologyOptimiser3DCompliance

__all__ = ["Domain3D", "EurocodeMaterialRegistry", "TopologyOptimiser3DCompliance"]
