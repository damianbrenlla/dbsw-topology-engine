# DBSW Core Package Initialisation
# Author: Damian Brenlla / DBSW 2026
#
# Works in two layouts:
#   1. Browser / Pyodide: python_core is mounted as the package name "core"
#      (solver_worker.js writes files under /home/pyodide/core/).
#   2. Local / CI: the package directory is named python_core; we still
#      expose the same public names so `from python_core import Domain3D`
#      and the test harness that aliases the folder as "core" both work.

try:
    # Preferred path when the package has been mounted / aliased as "core"
    from core.domain import Domain3D
    from core.materials import EurocodeMaterialRegistry
    from core.solvers import TopologyOptimiser3DCompliance
except ImportError:
    # Fallback for a normal install or direct import of python_core
    from .domain import Domain3D
    from .materials import EurocodeMaterialRegistry
    from .solvers import TopologyOptimiser3DCompliance

__all__ = ["Domain3D", "EurocodeMaterialRegistry", "TopologyOptimiser3DCompliance"]
