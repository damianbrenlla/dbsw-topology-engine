# DBSW 3D SIMP Topology Engine — Core Numerics
# Author: Damian Brenlla / DBSW 2026

import numpy as np
from scipy.signal import fftconvolve
import scipy.sparse as sp
from scipy.sparse.linalg import cg, spsolve
from core.domain import Domain3D


class TopologyOptimiser3DCompliance:
    """Deterministic 3D Hex8 SIMP Engine for Linear-Elastic Compliance Minimization

    Features:
      - Symmetric Isoparametric Integration Centered at Element Midpoints
      - Exact Cauchy Stress Sign Alignment (+Tension Top, -Compression Bottom)
      - Verified CG Solver Convergence with Direct-Solve Fallback
      - Physical (mm-based) Distance Filter Kernel
      - Sigmund (2007) Density-Weighted Sensitivity Filtering
    """

    def __init__(
        self,
        domain: Domain3D,
        volfrac: float = 0.3,
        penal_k: float = 3.0,
        rmin_mm: float = 150.0,
        notension_weight: float = 3.0,
    ):
        self.domain = domain
        self.volfrac = volfrac
        self.penal_k = penal_k
        self.rmin_mm = rmin_mm
        self.notension_weight = notension_weight

        self.Emax = float(domain.E0)  # MPa
        self.Emin = 1e-9 * self.Emax
        self.f_yd = float(domain.f_d)  # Design strength (MPa)

        self.x = np.full((domain.nx, domain.ny, domain.nz), volfrac)

        solid_mask = domain.passive_mask == 1.0
        void_mask = domain.passive_mask == 0.0
        self.x[solid_mask] = 1.0
        self.x[void_mask] = 0.001

        self.D_mat = self._build_constitutive_matrix()
        self.KE = self._build_element_stiffness()

        self._precompute_indexing()
        self._precompute_filter_kernel()

    def _build_constitutive_matrix(self):
        """6x6 physical isotropic elasticity tensor D in MPa."""
        E = self.Emax
        nu = self.domain.nu
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        D = c * np.array([
            [1 - nu, nu, nu, 0, 0, 0],
            [nu, 1 - nu, nu, 0, 0, 0],
            [nu, nu, 1 - nu, 0, 0, 0],
            [0, 0, 0, (1.0 - 2.0 * nu) / 2, 0, 0],
            [0, 0, 0, 0, (1.0 - 2.0 * nu) / 2, 0],
            [0, 0, 0, 0, 0, (1.0 - 2.0 * nu) / 2],
        ])
        return D

    def _get_shape_derivatives(self, xi, eta, zeta):
        """8x3 shape function derivatives dN/d(xi,eta,zeta) for standard Hex8."""
        return 0.125 * np.array([
            [-(1 - eta) * (1 - zeta), -(1 - xi) * (1 - zeta), -(1 - xi) * (1 - eta)],
            [(1 - eta) * (1 - zeta), -(1 + xi) * (1 - zeta), -(1 + xi) * (1 - eta)],
            [(1 + eta) * (1 - zeta), (1 + xi) * (1 - zeta), -(1 + xi) * (1 + eta)],
            [-(1 + eta) * (1 - zeta), (1 - xi) * (1 - zeta), -(1 - xi) * (1 + eta)],
            [-(1 - eta) * (1 + zeta), -(1 - xi) * (1 + zeta), (1 - xi) * (1 - eta)],
            [(1 - eta) * (1 + zeta), -(1 + xi) * (1 + zeta), (1 + xi) * (1 - eta)],
            [(1 + eta) * (1 + zeta), (1 + xi) * (1 + zeta), (1 + xi) * (1 + eta)],
            [-(1 + eta) * (1 + zeta), (1 - xi) * (1 + zeta), (1 - xi) * (1 + eta)],
        ])

    def _build_element_stiffness(self):
        """Integrates 24x24 element stiffness KE (N/mm) using 2x2x2 Gauss quadrature."""
        dx, dy, dz = self.domain.dx, self.domain.dy, self.domain.dz
        # Symmetric node coordinates centered at midpoints
        node_coords = np.array([
            [-0.5 * dx, -0.5 * dy, -0.5 * dz],
            [0.5 * dx, -0.5 * dy, -0.5 * dz],
            [0.5 * dx, 0.5 * dy, -0.5 * dz],
            [-0.5 * dx, 0.5 * dy, -0.5 * dz],
            [-0.5 * dx, -0.5 * dy, 0.5 * dz],
            [0.5 * dx, -0.5 * dy, 0.5 * dz],
            [0.5 * dx, 0.5 * dy, 0.5 * dz],
            [-0.5 * dx, 0.5 * dy, 0.5 * dz],
        ])

        gp = 1.0 / np.sqrt(3.0)
        gauss = [-gp, gp]
        KE = np.zeros((24, 24))

        for xi in gauss:
            for eta in gauss:
                for zeta in gauss:
                    dN_dxi = self._get_shape_derivatives(xi, eta, zeta)
                    J = dN_dxi.T @ node_coords
                    detJ = np.linalg.det(J)
                    invJ = np.linalg.inv(J)
                    dN_dx = dN_dxi @ invJ

                    B = np.zeros((6, 24))
                    for i in range(8):
                        B[0, 3 * i] = dN_dx[i, 0]
                        B[1, 3 * i + 1] = dN_dx[i, 1]
                        B[2, 3 * i + 2] = dN_dx[i, 2]
                        B[3, 3 * i] = dN_dx[i, 1]
                        B[3, 3 * i + 1] = dN_dx[i, 0]
                        B[4, 3 * i + 1] = dN_dx[i, 2]
                        B[4, 3 * i + 2] = dN_dx[i, 1]
                        B[5, 3 * i] = dN_dx[i, 2]
                        B[5, 3 * i + 2] = dN_dx[i, 0]

                    KE += detJ * (B.T @ self.D_mat @ B)

        return KE

    def evaluate_signed_element_stresses(self, stress_vec):
        """Calculates signed equivalent stress (MPa) indicating Tension (+) vs Compression (-)."""
        sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx = stress_vec

        S_tensor = np.array([
            [sig_x, tau_xy, tau_zx],
            [tau_xy, sig_y, tau_yz],
            [tau_zx, tau_yz, sig_z],
        ])

        p_vals = np.sort(np.linalg.eigvalsh(S_tensor))[::-1]
        sig1, sig3 = p_vals[0], p_vals[2]

        if self.domain.material_type in ["steel", "generic"]:
            vm = np.sqrt(
                0.5
                * (
                    (sig_x - sig_y) ** 2
                    + (sig_y - sig_z) ** 2
                    + (sig_z - sig_x) ** 2
                )
                + 3.0 * (tau_xy**2 + tau_yz**2 + tau_zx**2)
            )
            dominant_sig = sig1 if abs(sig1) >= abs(sig3) else sig3
            sign = 1.0 if dominant_sig >= 0 else -1.0
            return sign * vm
        else:
            return sig1 if abs(sig1) >= abs(sig3) else sig3

    def recover_element_stress_field(self, U):
        """Recovers physical element Cauchy stresses (MPa) evaluated at centroids."""
        dx, dy, dz = self.domain.dx, self.domain.dy, self.domain.dz
        node_coords = np.array([
            [-0.5 * dx, -0.5 * dy, -0.5 * dz],
            [0.5 * dx, -0.5 * dy, -0.5 * dz],
            [0.5 * dx, 0.5 * dy, -0.5 * dz],
            [-0.5 * dx, 0.5 * dy, -0.5 * dz],
            [-0.5 * dx, -0.5 * dy, 0.5 * dz],
            [0.5 * dx, -0.5 * dy, 0.5 * dz],
            [0.5 * dx, 0.5 * dy, 0.5 * dz],
            [-0.5 * dx, 0.5 * dy, 0.5 * dz],
        ])

        dN_dxi_0 = self._get_shape_derivatives(0.0, 0.0, 0.0)
        J_0 = dN_dxi_0.T @ node_coords
        invJ_0 = np.linalg.inv(J_0)
        dN_dx_0 = dN_dxi_0 @ invJ_0

        B_0 = np.zeros((6, 24))
        for i in range(8):
            B_0[0, 3 * i] = dN_dx_0[i, 0]
            B_0[1, 3 * i + 1] = dN_dx_0[i, 1]
            B_0[2, 3 * i + 2] = dN_dx_0[i, 2]
            B_0[3, 3 * i] = dN_dx_0[i, 1]
            B_0[3, 3 * i + 1] = dN_dx_0[i, 0]
            B_0[4, 3 * i + 1] = dN_dx_0[i, 2]
            B_0[4, 3 * i + 2] = dN_dx_0[i, 1]
            B_0[5, 3 * i] = dN_dx_0[i, 2]
            B_0[5, 3 * i + 2] = dN_dx_0[i, 0]

        U_elem = U[self.edof_vec]  # (Nelem, 24) in mm
        strains = U_elem @ B_0.T  # (Nelem, 6)
        raw_stresses = strains @ self.D_mat.T  # (Nelem, 6) in MPa

        elem_stresses_mpa = np.zeros(self.domain.n_elems)

        for e in range(self.domain.n_elems):
            stress_vec = raw_stresses[e, :]
            elem_stresses_mpa[e] = self.evaluate_signed_element_stresses(
                stress_vec
            )

        return elem_stresses_mpa

    def recover_nodal_displacements(self, U):
        """Extracts 3D displacement vectors [Ux, Uy, Uz] and magnitudes (mm) for all grid nodes."""
        U_nodes = U.reshape((-1, 3))
        disp_magnitudes = np.linalg.norm(U_nodes, axis=1)
        return U_nodes, disp_magnitudes

    def _precompute_indexing(self):
        nx, ny, nz = self.domain.nx, self.domain.ny, self.domain.nz
        nodenrs = np.arange(self.domain.n_nodes).reshape(
            (nx + 1, ny + 1, nz + 1)
        )

        edof_vec = np.zeros((self.domain.n_elems, 24), dtype=int)
        el = 0
        for ex in range(nx):
            for ey in range(ny):
                for ez in range(nz):
                    nodes = [
                        nodenrs[ex, ey, ez],
                        nodenrs[ex + 1, ey, ez],
                        nodenrs[ex + 1, ey + 1, ez],
                        nodenrs[ex, ey + 1, ez],
                        nodenrs[ex, ey, ez + 1],
                        nodenrs[ex + 1, ey, ez + 1],
                        nodenrs[ex + 1, ey + 1, ez + 1],
                        nodenrs[ex, ey + 1, ez + 1],
                    ]
                    edof = []
                    for n in nodes:
                        edof.extend([3 * n, 3 * n + 1, 3 * n + 2])
                    edof_vec[el, :] = edof
                    el += 1

        self.edof_vec = edof_vec
        self.iK = np.kron(edof_vec, np.ones((24, 1), dtype=int)).flatten()
        self.jK = np.kron(edof_vec, np.ones((1, 24), dtype=int)).flatten()

    def _precompute_filter_kernel(self):
        """Precomputes physical distance-based filter kernel in mm."""
        rx_vox = int(np.ceil(self.rmin_mm / self.domain.dx))
        ry_vox = int(np.ceil(self.rmin_mm / self.domain.dy))
        rz_vox = int(np.ceil(self.rmin_mm / self.domain.dz))

        px, py, pz = np.meshgrid(
            np.arange(-rx_vox, rx_vox + 1),
            np.arange(-ry_vox, ry_vox + 1),
            np.arange(-rz_vox, rz_vox + 1),
            indexing="ij",
        )

        dist_mm = np.sqrt(
            (px * self.domain.dx) ** 2 +
            (py * self.domain.dy) ** 2 +
            (pz * self.domain.dz) ** 2
        )

        self.filter_kernel = np.maximum(0.0, self.rmin_mm - dist_mm)

        ones_grid = np.ones((self.domain.nx, self.domain.ny, self.domain.nz))
        self.filter_Hsum = fftconvolve(ones_grid, self.filter_kernel, mode="same")
        self.filter_Hsum[self.filter_Hsum <= 0] = 1.0

    def filter_sensitivity(self, dc, x):
        """Sigmund (2007) density-weighted sensitivity filter."""
        numerator = fftconvolve(x * dc, self.filter_kernel, mode="same")
        denom = np.maximum(x, 1e-3) * self.filter_Hsum
        return numerator / denom

    def compute_self_weight_vector(self, x):
        """Computes self-weight body force vector in -Z (N)."""
        ndof = self.domain.ndof
        Fsw = np.zeros(ndof)

        if self.domain.gamma_n_mm3 <= 0.0:
            return Fsw

        vol_elem = self.domain.dx * self.domain.dy * self.domain.dz
        x_flat = x.flatten()
        elem_weight_n = self.domain.gamma_n_mm3 * vol_elem * x_flat
        node_weight_n = elem_weight_n / 8.0

        z_dofs = self.edof_vec[:, 2::3]
        np.add.at(Fsw, z_dofs.flatten(), np.repeat(-node_weight_n, 8))

        return Fsw

    def compute_notension_penalty(self, U):
        """Approximate Rankine sensitivity penalty for brittle materials."""
        elem_stress = self.recover_element_stress_field(U)
        stress_grid = elem_stress.reshape(
            (self.domain.nx, self.domain.ny, self.domain.nz)
        )
        tension_part_mpa = np.maximum(stress_grid, 0.0)
        penalty = (
            self.notension_weight
            * (tension_part_mpa / max(self.f_yd, 1e-6))
            * (self.Emax - self.Emin)
        )
        return penalty

    def assemble_and_solve_static(self, x, include_self_weight=False):
        """Assembles stiffness matrix and solves KU = F with CG verification & direct fallback."""
        ndof = self.domain.ndof
        x_flat = x.flatten()

        E_factor = (self.Emin / self.Emax) + (x_flat**self.penal_k) * (
            1.0 - (self.Emin / self.Emax)
        )
        sK = (
            self.KE.flatten()[np.newaxis, :] * E_factor[:, np.newaxis]
        ).flatten()

        K = sp.coo_matrix((sK, (self.iK, self.jK)), shape=(ndof, ndof)).tocsr()

        F = self.domain.F.copy()
        if include_self_weight:
            F = F + self.compute_self_weight_vector(x)

        fixed = np.unique(self.domain.fixed_dofs)
        free = np.setdiff1d(np.arange(ndof), fixed)

        K_free = K[free, :][:, free]
        F_free = F[free]

        diag_K = K_free.diagonal()
        diag_K[diag_K == 0] = 1.0
        M = sp.coo_matrix(
            (1.0 / diag_K, (np.arange(len(free)), np.arange(len(free))))
        ).tocsr()

        u_free, info = cg(
            K_free, F_free, M=M, maxiter=3000, tol=1e-8
        )

        if info != 0:
            u_free = spsolve(K_free, F_free)

        U = np.zeros(ndof)
        U[free] = u_free
        return U, K, free
