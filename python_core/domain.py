"""
DBSW R260003 Topology Optimiser - Continuum Domain Module
Domain discretization, 3D Hex8 FE topology formulation, and passive element masks.
"""

import numpy as np


class Domain3D:
    def __init__(self, nelx: int, nely: int, nelz: int, lx: float, ly: float, lz: float):
        """
        Defines a 3D rectangular continuum domain discretized with Hex8 elements.
        
        Parameters:
        - nelx, nely, nelz: Element counts along X, Y, Z axes
        - lx, ly, lz: Domain physical dimensions in meters
        """
        self.nelx = int(nelx)
        self.nely = int(nely)
        self.nelz = int(nelz)
        self.num_elements = self.nelx * self.nely * self.nelz
        
        self.lx = float(lx)
        self.ly = float(ly)
        self.lz = float(lz)
        
        self.dx = self.lx / self.nelx
        self.dy = self.ly / self.nely
        self.dz = self.lz / self.nelz
        
        # Total nodes across 3D grid
        self.nnx = self.nelx + 1
        self.nny = self.nely + 1
        self.nnz = self.nelz + 1
        self.num_nodes = self.nnx * self.nny * self.nnz
        self.dof_per_node = 3
        self.num_dofs = self.num_nodes * self.dof_per_node
        
        # Passive element masks: 1 = designable, 0 = passive solid, -1 = passive void
        self.passive_mask = np.ones((self.nelx, self.nely, self.nelz), dtype=np.int8)

    def set_passive_solid(self, x_bounds: tuple, y_bounds: tuple, z_bounds: tuple):
        """Enforces non-optimisable solid zones (e.g. bearing plates, stiffener regions)."""
        x0, x1 = self._clamp_indices(x_bounds, self.nelx)
        y0, y1 = self._clamp_indices(y_bounds, self.nely)
        z0, z1 = self._clamp_indices(z_bounds, self.nelz)
        self.passive_mask[x0:x1, y0:y1, z0:z1] = 0

    def set_passive_void(self, x_bounds: tuple, y_bounds: tuple, z_bounds: tuple):
        """Enforces non-optimisable void zones (e.g. service penetrations, keep-outs)."""
        x0, x1 = self._clamp_indices(x_bounds, self.nelx)
        y0, y1 = self._clamp_indices(y_bounds, self.nely)
        z0, z1 = self._clamp_indices(z_bounds, self.nelz)
        self.passive_mask[x0:x1, y0:y1, z0:z1] = -1

    def _clamp_indices(self, bounds: tuple, max_val: int) -> tuple:
        i0 = max(0, min(int(bounds[0]), max_val))
        i1 = max(0, min(int(bounds[1]), max_val))
        return min(i0, i1), max(i0, i1)

    def get_element_dofs(self) -> np.ndarray:
        """
        Precomputes element-to-node DOF mapping for Hex8 elements.
        Standard 24-DOF connectivity array for sparse global stiffness assembly.
        """
        edof = np.zeros((self.num_elements, 24), dtype=np.int32)
        idx = 0
        for elz in range(self.nelz):
            for ely in range(self.nely):
                for elx in range(self.nelx):
                    # Node indices for current Hex8 element
                    n1 = elz * (self.nnx * self.nny) + ely * self.nnx + elx
                    n2 = n1 + 1
                    n3 = n1 + self.nnx + 1
                    n4 = n1 + self.nnx
                    n5 = n1 + (self.nnx * self.nny)
                    n6 = n2 + (self.nnx * self.nny)
                    n7 = n3 + (self.nnx * self.nny)
                    n8 = n4 + (self.nnx * self.nny)
                    
                    nodes = [n1, n2, n3, n4, n5, n6, n7, n8]
                    dofs = []
                    for n in nodes:
                        dofs.extend([3 * n, 3 * n + 1, 3 * n + 2])
                    
                    edof[idx, :] = dofs
                    idx += 1
        return edof
