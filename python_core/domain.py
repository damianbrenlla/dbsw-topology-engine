# DBSW 3D Spatial Domain Definition Engine
# Author: Damian Brenlla / DBSW 2026
# v2 — Added add_line_support(). Previously, 3D line supports were built in
#      solver_worker.js by sampling points along the line and calling
#      add_support_box() with a small isotropic search-box radius at each
#      sample. add_support_box() has no real node-finding tolerance of its
#      own (eps=1e-2mm, meant only to absorb floating-point rounding) — so
#      whether a sample actually captured a node depended entirely on that
#      worker-side radius being large enough on EVERY axis. For an
#      anisotropic mesh (dx != dy != dz, which is the norm for a wall/beam
#      domain like Lx=6000 Ly=300 Lz=600 with nx=36 ny=9 nz=12), a radius
#      sized off the mesh's finest spacing is too small on the coarser axes.
#      Confirmed by direct test: a line support placed at a Y coordinate
#      more than ~12.5mm from an existing node row (a ~21mm-wide gap out of
#      every 33.3mm between node rows, on the default domain) captured
#      ZERO degrees of freedom — silently. The structure would then solve
#      with less restraint than intended, or fail outright if that was the
#      only support defined.
#
#      add_line_support() below sidesteps the box-radius problem entirely by
#      reusing the same nearest-node INDEX ROUNDING already proven correct
#      in add_point_load() (round(coord / spacing), clipped to the node
#      range) rather than testing whether a node falls inside a small box.
#      That guarantees exactly one node is captured per sample, on every
#      axis, regardless of how anisotropic dx/dy/dz are.

import numpy as np


class Domain3D:
    """3D Spatial Continuum Domain for FEA and Topology Optimization."""

    def __init__(
        self,
        Lx: float,
        Ly: float,
        Lz: float,
        nx: int,
        ny: int,
        nz: int,
        E: float = 210000.0,
        nu: float = 0.30,
        f_k: float = 355.0,
        f_d: float = 355.0,
        material_type: str = "steel",
        material_name: str = "Steel S355",
        gamma_kn_m3: float = 0.0,
    ):
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.Lz = float(Lz)
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)

        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz

        self.n_elems = self.nx * self.ny * self.nz
        self.n_nodes = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
        self.ndof = 3 * self.n_nodes

        self.E0 = float(E)
        self.nu = float(nu)
        self.f_k = float(f_k)
        self.f_d = float(f_d)

        allowed_materials = ["steel", "concrete", "timber", "masonry", "stone", "generic"]
        mat_lower = material_type.lower()
        if mat_lower not in allowed_materials:
            raise ValueError(f"Unsupported material_type '{material_type}'. Must be one of {allowed_materials}")

        self.material_type = mat_lower
        self.material_name = material_name

        # Unit weight: 1 kN/m3 = 1e-6 N/mm3
        self.gamma_kn_m3 = float(gamma_kn_m3)
        self.gamma_n_mm3 = self.gamma_kn_m3 * 1e-6

        self.F = np.zeros(self.ndof)
        self.fixed_dofs = []

        # Passive region mask: 1.0 = forced solid, 0.0 = forced void, -1.0 = free design space
        self.passive_mask = np.full((self.nx, self.ny, self.nz), -1.0)

    def _get_node_grid(self):
        """Canonical 3D Node Grid in (nx+1, ny+1, nz+1) shape."""
        return np.arange(self.n_nodes).reshape((self.nx + 1, self.ny + 1, self.nz + 1))

    def add_support_box(self, x_bounds, y_bounds, z_bounds, dofs="xyz"):
        """Restrains DOFs and protects surrounding elements as solid passive mask."""
        x_min, x_max = min(x_bounds), max(x_bounds)
        y_min, y_max = min(y_bounds), max(y_bounds)
        z_min, z_max = min(z_bounds), max(z_bounds)

        nodenrs = self._get_node_grid()
        eps = 1e-2

        for ix in range(self.nx + 1):
            px = ix * self.dx
            if x_min - eps <= px <= x_max + eps:
                for iy in range(self.ny + 1):
                    py = iy * self.dy
                    if y_min - eps <= py <= y_max + eps:
                        for iz in range(self.nz + 1):
                            pz = iz * self.dz
                            if z_min - eps <= pz <= z_max + eps:
                                node = nodenrs[ix, iy, iz]
                                if "x" in dofs:
                                    self.fixed_dofs.append(3 * node)
                                if "y" in dofs:
                                    self.fixed_dofs.append(3 * node + 1)
                                if "z" in dofs:
                                    self.fixed_dofs.append(3 * node + 2)

                                ex = min(ix, self.nx - 1)
                                ey = min(iy, self.ny - 1)
                                ez = min(iz, self.nz - 1)
                                self.passive_mask[ex, ey, ez] = 1.0

        self.fixed_dofs = list(set(self.fixed_dofs))

    def add_line_support(self, p1_xyz, p2_xyz, dofs="xyz"):
        """
        Restrains DOFs for every mesh node lying along the 3D line segment
        from p1_xyz to p2_xyz.

        FIX (see module docstring): rather than testing whether nodes fall
        inside a small search box around sample points along the line (which
        silently captures nothing if the box is smaller than the local mesh
        spacing on any axis -- a real, confirmed failure mode for
        anisotropic meshes), this samples t along the line and snaps each
        sample to its NEAREST node index per axis via
        round(coord / spacing), exactly mirroring add_point_load()'s proven
        approach. That guarantees a node is always captured on every axis,
        independent of dx/dy/dz being unequal.

        Args:
            p1_xyz, p2_xyz: [x, y, z] endpoints of the support line, in mm.
            dofs: string containing any of 'x', 'y', 'z' -- which
                translational DOFs to restrain at each captured node.
        """
        p1 = np.array(p1_xyz, dtype=float)
        p2 = np.array(p2_xyz, dtype=float)
        vec = p2 - p1
        length = float(np.linalg.norm(vec))

        nodenrs = self._get_node_grid()

        if length < 1e-6:
            # Degenerate (zero-length) line -- treat as a single point.
            candidates = [p1]
        else:
            # Sample finely enough that consecutive samples can never skip a
            # node on whichever axis the line is changing fastest along.
            # Using the mesh's FINEST spacing as the step is always safe
            # (it can only ever over-sample, never under-sample, regardless
            # of the line's direction) -- and because capture below is now
            # index-based rather than box-based, redundant samples just
            # resolve to the same node index and get deduplicated by the
            # `seen_nodes` set. (This isn't necessarily faster than the old
            # box-sampling for a single line in isolation -- the win here is
            # correctness on anisotropic meshes, not raw speed.)
            step = min(self.dx, self.dy, self.dz) * 0.5
            n_samples = max(2, int(np.ceil(length / step)) + 1)
            t_vals = np.linspace(0.0, 1.0, n_samples)
            candidates = [p1 + t * vec for t in t_vals]

        seen_nodes = set()
        for pt in candidates:
            ix = int(np.clip(round(pt[0] / self.dx), 0, self.nx))
            iy = int(np.clip(round(pt[1] / self.dy), 0, self.ny))
            iz = int(np.clip(round(pt[2] / self.dz), 0, self.nz))
            seen_nodes.add((ix, iy, iz))

        for ix, iy, iz in seen_nodes:
            node = nodenrs[ix, iy, iz]
            if "x" in dofs:
                self.fixed_dofs.append(3 * node)
            if "y" in dofs:
                self.fixed_dofs.append(3 * node + 1)
            if "z" in dofs:
                self.fixed_dofs.append(3 * node + 2)

            ex = min(ix, self.nx - 1)
            ey = min(iy, self.ny - 1)
            ez = min(iz, self.nz - 1)
            self.passive_mask[ex, ey, ez] = 1.0

        self.fixed_dofs = list(set(self.fixed_dofs))

    def add_point_load(self, coord_xyz, force_xyz):
        """Applies point force [Fx, Fy, Fz] in N and protects surrounding elements as solid."""
        px, py, pz = coord_xyz
        ix = int(np.clip(round(px / self.dx), 0, self.nx))
        iy = int(np.clip(round(py / self.dy), 0, self.ny))
        iz = int(np.clip(round(pz / self.dz), 0, self.nz))

        nodenrs = self._get_node_grid()
        node = nodenrs[ix, iy, iz]

        self.F[3 * node] += force_xyz[0]
        self.F[3 * node + 1] += force_xyz[1]
        self.F[3 * node + 2] += force_xyz[2]

        ex = min(ix, self.nx - 1)
        ey = min(iy, self.ny - 1)
        ez = min(iz, self.nz - 1)
        self.passive_mask[ex, ey, ez] = 1.0

    def add_patch_load(self, x_bounds, y_bounds, z_bounds, total_load_xyz):
        """Distributes patch load uniformly across all nodes within bounding region."""
        x_min, x_max = min(x_bounds), max(x_bounds)
        y_min, y_max = min(y_bounds), max(y_bounds)
        z_min, z_max = min(z_bounds), max(z_bounds)

        nodenrs = self._get_node_grid()
        eps = 1e-2

        target_nodes = []
        for ix in range(self.nx + 1):
            px = ix * self.dx
            if x_min - eps <= px <= x_max + eps:
                for iy in range(self.ny + 1):
                    py = iy * self.dy
                    if y_min - eps <= py <= y_max + eps:
                        for iz in range(self.nz + 1):
                            pz = iz * self.dz
                            if z_min - eps <= pz <= z_max + eps:
                                target_nodes.append((ix, iy, iz, nodenrs[ix, iy, iz]))

        if len(target_nodes) > 0:
            load_per_node = np.array(total_load_xyz) / len(target_nodes)
            for ix, iy, iz, node in target_nodes:
                self.F[3 * node] += load_per_node[0]
                self.F[3 * node + 1] += load_per_node[1]
                self.F[3 * node + 2] += load_per_node[2]

                ex = min(ix, self.nx - 1)
                ey = min(iy, self.ny - 1)
                ez = min(iz, self.nz - 1)
                self.passive_mask[ex, ey, ez] = 1.0
