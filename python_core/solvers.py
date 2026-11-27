# DBSW 3D SIMP Topology Engine — Core Numerics
# Author: Damian Brenlla / DBSW 2026

import numpy as np
from scipy.signal import fftconvolve
import scipy.sparse as sp
from scipy.sparse.linalg import cg, spsolve
from core.domain import Domain3D


class TopologyOptimiser3DCompliance:
    """Deterministic 3D Hex8 SIMP Engine for Linear-Elastic Compliance Minimization"""

    def __init__(
        self,
        domain: Domain3D,
        volfrac: float = 0.3,
        penal_k: float = 3.0,
        rmin_mm: float = 150.0,
        notension_weight: float = 3.0,
    ):
        self.domain = domain
        self.volfrac = float(volfrac)
        if not np.isfinite(self.volfrac) or not (0.0 < self.volfrac <= 1.0):
            raise ValueError("Volume fraction must be greater than 0 and no greater than 1.")
        self.penal_k = float(penal_k)
        self.requested_rmin_mm = float(rmin_mm)
        if not np.isfinite(self.requested_rmin_mm) or self.requested_rmin_mm <= 0.0:
            raise ValueError("Sensitivity filter radius must be a finite positive value.")
        # A filter radius smaller than the coarsest element spacing excludes
        # the adjacent element in that direction entirely. Use a meaningful
        # physical radius (1.5 element widths) rather than silently running
        # an effectively unfiltered optimisation in that direction.
        self.minimum_rmin_mm = 1.5 * max(domain.dx, domain.dy, domain.dz)
        self.rmin_mm = max(self.requested_rmin_mm, self.minimum_rmin_mm)
        self.notension_weight = float(notension_weight)

        self.Emax = float(domain.E0)  # MPa
        self.Emin = 1e-9 * self.Emax
        self.f_yd = float(domain.f_d)  # Design strength (MPa)

        # Fix: Start baseline analysis domain at 1.0 (100% solid)
        self.x = np.ones((domain.nx, domain.ny, domain.nz))

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
        node_coords = np.array([
            [0.0, 0.0, 0.0],
            [dx, 0.0, 0.0],
            [dx, dy, 0.0],
            [0.0, dy, 0.0],
            [0.0, 0.0, dz],
            [dx, 0.0, dz],
            [dx, dy, dz],
            [0.0, dy, dz],
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
            [0.0, 0.0, 0.0],
            [dx, 0.0, 0.0],
            [dx, dy, 0.0],
            [0.0, dy, 0.0],
            [0.0, 0.0, dz],
            [dx, 0.0, dz],
            [dx, dy, dz],
            [0.0, dy, dz],
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
        numerator = fftconvolve(x * dc, self.filter_kernel, mode="same")
        denom = np.maximum(x, 1e-3) * self.filter_Hsum
        return numerator / denom

    def compute_self_weight_vector(self, x):
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

    def compute_compliance_sensitivity(self, U, x, include_self_weight=False):
        """Return dC/dx for compliance, including density-dependent self-weight.

        For C = F(x)^T U and K(x)U = F(x), the derivative is
        2 U^T dF/dx - U^T dK/dx U. The stiffness term is the standard SIMP
        energy derivative; the first term is required when self-weight is
        included because the gravity load is proportional to element density.
        """
        x = np.asarray(x, dtype=float)
        U_elem = U[self.edof_vec]
        element_energy = np.sum((U_elem @ self.KE) * U_elem, axis=1)
        dc = (
            -self.penal_k
            * np.power(np.maximum(x.reshape(-1), 1e-3), self.penal_k - 1.0)
            * (self.Emax - self.Emin)
            * element_energy
        )
        if include_self_weight and self.domain.gamma_n_mm3 > 0.0:
            element_volume = self.domain.dx * self.domain.dy * self.domain.dz
            nodal_dFz_dx = -self.domain.gamma_n_mm3 * element_volume / 8.0
            dC_load = 2.0 * nodal_dFz_dx * np.sum(U_elem[:, 2::3], axis=1)
            dc = dc + dC_load
        return dc.reshape(x.shape)

    def optimality_criteria_update(self, dc, move_limit=0.2, max_iterations=100):
        """Bounded OC update with an adaptive multiplier bracket."""
        x = np.asarray(self.x, dtype=float)
        passive_solid = self.domain.passive_mask == 1.0
        passive_void = self.domain.passive_mask == 0.0
        free = ~(passive_solid | passive_void)
        target_volume = float(self.volfrac * x.size)
        fixed_volume = float(np.count_nonzero(passive_solid)) + 0.001 * float(np.count_nonzero(passive_void))
        if fixed_volume > target_volume + 1e-8:
            raise ValueError("Volume fraction is infeasible: forced-solid and passive-void regions alone exceed the requested material volume.")
        if not np.any(free):
            return x.copy()

        move = float(move_limit)
        if not np.isfinite(move) or not (0.0 < move <= 1.0):
            raise ValueError("OC move limit must be greater than 0 and no greater than 1.")

        lower = np.maximum(0.001, x - move)
        upper = np.minimum(1.0, x + move)
        lower[passive_solid] = upper[passive_solid] = 1.0
        lower[passive_void] = upper[passive_void] = 0.001

        # If the current iterate and move bound make the target unreachable,
        # relax only the lower move bound enough to restore feasibility.
        if float(np.sum(lower)) > target_volume + 1e-8:
            available = target_volume - fixed_volume
            nfree = int(np.count_nonzero(free))
            if available < 0.001 * nfree - 1e-8:
                raise ValueError("Volume fraction is infeasible: passive regions leave less than the minimum free-element density.")
            old_free_lower = lower[free] - 0.001
            old_total = float(np.sum(old_free_lower))
            scale = 0.0 if old_total <= 0.0 else max(0.0, min(1.0, (available - 0.001 * nfree) / old_total))
            lower[free] = 0.001 + old_free_lower * scale

        dc_flat = np.asarray(dc, dtype=float).reshape(x.shape)
        if not np.all(np.isfinite(dc_flat)):
            raise ValueError("OC sensitivities contain non-finite values.")
        ratio_numerator = np.maximum(-dc_flat, 1e-30)

        def candidate_for(multiplier):
            ratio = np.sqrt(ratio_numerator / max(float(multiplier), 1e-300))
            candidate = np.minimum(upper, np.maximum(lower, x * ratio))
            candidate[passive_solid] = 1.0
            candidate[passive_void] = 0.001
            return candidate

        # Adapt the upper bracket to the actual sensitivity magnitude.
        multiplier_low = 0.0
        multiplier_high = max(float(np.max(ratio_numerator)), 1.0)
        candidate = candidate_for(multiplier_high)
        for _ in range(200):
            if float(np.sum(candidate)) <= target_volume:
                break
            multiplier_high *= 2.0
            if not np.isfinite(multiplier_high):
                raise RuntimeError("Could not bracket a feasible OC multiplier.")
            candidate = candidate_for(multiplier_high)
        else:
            raise RuntimeError("Could not find a feasible OC multiplier for the requested volume.")

        best_feasible = candidate
        for _ in range(max(1, int(max_iterations))):
            mid = 0.5 * (multiplier_low + multiplier_high)
            candidate = candidate_for(mid)
            if float(np.sum(candidate)) > target_volume:
                multiplier_low = mid
            else:
                multiplier_high = mid
                best_feasible = candidate
            if multiplier_high > 0.0 and (multiplier_high - multiplier_low) / multiplier_high <= 1e-8:
                break

        best_feasible[passive_solid] = 1.0
        best_feasible[passive_void] = 0.001
        if float(np.sum(best_feasible)) > target_volume + 1e-6:
            raise RuntimeError("OC update remains infeasible after adaptive bracketing; check passive regions and the requested volume fraction.")
        return best_feasible

    def compute_notension_penalty(self, U):
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

        try:
            u_free, info = cg(
                K_free, F_free, M=M, maxiter=3000, rtol=1e-5, atol=1e-8
            )
        except TypeError:
            u_free, info = cg(
                K_free, F_free, M=M, maxiter=3000, tol=1e-5, atol=1e-8
            )

        if info != 0:
            print(f"[Solver Warning]: SciPy CG solver failed to converge (info={info}). Falling back to spsolve direct solver.")
            u_free = spsolve(K_free, F_free)

        U = np.zeros(ndof)
        U[free] = u_free
        return U, K, free
