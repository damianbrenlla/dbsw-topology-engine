"""Regression tests for R260004 topology optimisation safeguards.

Run with: python -m unittest discover -s tests -v
"""
import os
import sys
import types
import unittest
import numpy as np

CORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python_core"))
sys.path.insert(0, CORE_PATH)
# The browser mounts python_core as /home/pyodide/core; mirror that package
# name for local/CI tests without changing the deployed module layout.
core_package = types.ModuleType("core")
core_package.__path__ = [CORE_PATH]
sys.modules.setdefault("core", core_package)

from core.domain import Domain3D
from core.solvers import TopologyOptimiser3DCompliance


def make_domain(nx=3, ny=2, nz=2):
    return Domain3D(
        Lx=600.0, Ly=200.0, Lz=200.0,
        nx=nx, ny=ny, nz=nz,
        E=30000.0, nu=0.2, f_k=30.0, f_d=20.0,
        material_type="concrete", material_name="Test concrete",
        gamma_kn_m3=25.0,
    )


class SolverSafeguardTests(unittest.TestCase):
    def test_filter_reaches_neighbour_on_coarsest_axis(self):
        domain = Domain3D(
            Lx=6000.0, Ly=300.0, Lz=600.0,
            nx=36, ny=9, nz=12,
            E=30000.0, nu=0.2, f_k=30.0, f_d=20.0,
            material_type="concrete", material_name="Test concrete",
            gamma_kn_m3=25.0,
        )
        opt = TopologyOptimiser3DCompliance(domain, volfrac=0.5, rmin_mm=150.0)
        self.assertGreaterEqual(opt.rmin_mm, 1.5 * max(domain.dx, domain.dy, domain.dz))
        kernel = opt.filter_kernel
        cx, cy, cz = (n // 2 for n in kernel.shape)
        self.assertGreater(kernel[cx + 1, cy, cz], 0.0)

    def test_self_weight_sensitivity_contains_load_derivative(self):
        domain = make_domain()
        opt = TopologyOptimiser3DCompliance(domain, volfrac=0.5, rmin_mm=300.0)
        x = np.full((domain.nx, domain.ny, domain.nz), 0.5)
        U = np.linspace(-0.02, 0.03, domain.ndof)
        with_sw = opt.compute_compliance_sensitivity(U, x, include_self_weight=True)
        without_sw = opt.compute_compliance_sensitivity(U, x, include_self_weight=False)
        self.assertTrue(np.all(np.isfinite(with_sw)))
        self.assertGreater(float(np.linalg.norm(with_sw - without_sw)), 0.0)

    def test_penalisation_continuation_is_smooth_and_reaches_endpoints(self):
        values = [
            TopologyOptimiser3DCompliance.continuation_penalisation(i, 20, 1.0, 3.0, 0.5)
            for i in range(20)
        ]
        self.assertAlmostEqual(values[0], 1.0)
        self.assertAlmostEqual(values[-1], 3.0)
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))
        # No discrete 0.5 jump: the maximum per-iteration increment is small.
        self.assertLess(max(b - a for a, b in zip(values, values[1:])), 0.2)

    def test_no_tension_penalty_is_scaled_and_filtered(self):
        domain = make_domain()
        opt = TopologyOptimiser3DCompliance(domain, volfrac=0.5, rmin_mm=300.0)
        x = np.full((domain.nx, domain.ny, domain.nz), 0.5)
        base = -np.full_like(x, 100.0)
        penalty = np.full_like(x, 15000.0)
        combined = opt.combine_notension_sensitivity(base, penalty, x)
        self.assertTrue(np.all(np.isfinite(combined)))
        self.assertTrue(np.all(combined > base))
        self.assertTrue(np.all(combined < 0.0))

    def test_oc_preserves_passive_regions_and_volume(self):
        domain = make_domain()
        domain.passive_mask[0, 0, 0] = 1.0
        domain.passive_mask[-1, -1, -1] = 0.0
        opt = TopologyOptimiser3DCompliance(domain, volfrac=0.5, rmin_mm=300.0)
        opt.x[:] = 0.5
        dc = -np.ones_like(opt.x)
        updated = opt.optimality_criteria_update(dc, move_limit=0.2)
        self.assertEqual(updated[0, 0, 0], 1.0)
        self.assertEqual(updated[-1, -1, -1], 0.001)
        self.assertLessEqual(float(np.sum(updated)), opt.volfrac * updated.size + 1e-4)

    def test_oc_rejects_infeasible_passive_solid_volume(self):
        domain = make_domain()
        domain.passive_mask[:] = 1.0
        opt = TopologyOptimiser3DCompliance(domain, volfrac=0.5, rmin_mm=300.0)
        with self.assertRaisesRegex(ValueError, "forced-solid"):
            opt.optimality_criteria_update(-np.ones_like(opt.x))

    def test_oc_adaptive_bracket_handles_large_sensitivity_scale(self):
        domain = make_domain()
        opt = TopologyOptimiser3DCompliance(domain, volfrac=0.5, rmin_mm=300.0)
        opt.x[:] = 0.5
        updated = opt.optimality_criteria_update(-np.full_like(opt.x, 1e20), move_limit=0.2)
        self.assertTrue(np.all(np.isfinite(updated)))
        self.assertLessEqual(float(np.sum(updated)), opt.volfrac * updated.size + 1e-6)

    def test_oc_relaxes_move_lower_bound_when_passive_solid_makes_it_infeasible(self):
        domain = make_domain()
        domain.passive_mask[0, 0, 0] = 1.0
        opt = TopologyOptimiser3DCompliance(domain, volfrac=0.2, rmin_mm=300.0)
        opt.x[:] = 0.2
        opt.x[domain.passive_mask == 1.0] = 1.0
        updated = opt.optimality_criteria_update(-np.ones_like(opt.x), move_limit=0.05)
        self.assertEqual(updated[0, 0, 0], 1.0)
        self.assertLessEqual(float(np.sum(updated)), opt.volfrac * updated.size + 1e-6)

    def test_filter_rejects_invalid_radius(self):
        domain = make_domain()
        with self.assertRaisesRegex(ValueError, "filter radius"):
            TopologyOptimiser3DCompliance(domain, volfrac=0.5, rmin_mm=0.0)

    def test_point_support_snaps_to_nearest_node_on_anisotropic_mesh(self):
        domain = Domain3D(
            Lx=6000.0, Ly=300.0, Lz=600.0,
            nx=36, ny=9, nz=12,
        )
        domain.add_point_support([100.0, 16.0, 24.0], dofs="xyz")
        self.assertEqual(len(domain.fixed_dofs), 3)
        self.assertEqual(int(np.count_nonzero(domain.passive_mask == 1.0)), 1)

    def test_support_box_reports_when_no_node_is_captured(self):
        domain = make_domain()
        with self.assertRaisesRegex(ValueError, "did not capture"):
            domain.add_support_box([1.0, 2.0], [1.0, 2.0], [1.0, 2.0])

    def test_patch_load_reports_when_no_node_is_captured(self):
        domain = make_domain()
        with self.assertRaisesRegex(ValueError, "did not capture"):
            domain.add_patch_load([1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [0.0, 0.0, -1000.0])


class ReferenceComplianceTests(unittest.TestCase):
    """Small fixed-mesh compliance checks against a pure solid baseline.

    These are not published literature benchmarks (meshes are deliberately
    coarse so the suite stays fast). They assert that:
      - a fully solid domain under a known load produces a finite, positive
        compliance of a stable order of magnitude;
      - introducing a volume-constrained SIMP step reduces the mean density
        toward the target while compliance remains finite and positive;
      - self-weight, when enabled, increases the recovered compliance.
    """

    def _cantilever_domain(self, nx=8, ny=2, nz=2):
        """Short steel cantilever: fixed at x=0, tip load at x=Lx."""
        Lx, Ly, Lz = 400.0, 100.0, 100.0
        domain = Domain3D(
            Lx=Lx, Ly=Ly, Lz=Lz,
            nx=nx, ny=ny, nz=nz,
            E=210000.0, nu=0.30, f_k=355.0, f_d=355.0,
            material_type="steel", material_name="Steel S355",
            gamma_kn_m3=78.5,
        )
        domain.add_support_box([0.0, 0.0], [0.0, Ly], [0.0, Lz], dofs="xyz")
        # 1 kN downward at the free-end centre
        domain.add_point_load([Lx, Ly / 2.0, Lz / 2.0], [0.0, 0.0, -1000.0])
        return domain

    def test_solid_cantilever_compliance_is_finite_and_positive(self):
        domain = self._cantilever_domain()
        opt = TopologyOptimiser3DCompliance(
            domain, volfrac=1.0, penal_k=1.0, rmin_mm=80.0
        )
        # Force a fully solid density field
        opt.x[:] = 1.0
        U, _, _ = opt.assemble_and_solve_static(opt.x, include_self_weight=False)
        self.assertTrue(np.all(np.isfinite(U)))
        # Compliance C = U^T F  (self-weight off, so F is the applied load only)
        C = float(U @ domain.F)
        self.assertGreater(C, 0.0)
        self.assertTrue(np.isfinite(C))
        # Order-of-magnitude sanity for this coarse mesh & 1 kN tip load.
        # Analytical Euler-Bernoulli tip deflection is ~0.03 mm → C ~ 0.015 N·mm;
        # the coarse Hex8 mesh (only 2 elements through depth) is much softer
        # because of shear locking / under-integration of bending, so C is
        # larger. Bound is deliberately loose — the point is finiteness and
        # a stable physical order of magnitude, not a literature benchmark.
        self.assertLess(C, 50.0)
        self.assertGreater(C, 1e-4)

    def test_one_oc_step_reduces_mean_density_toward_volfrac(self):
        domain = self._cantilever_domain()
        volfrac = 0.4
        opt = TopologyOptimiser3DCompliance(
            domain, volfrac=volfrac, penal_k=3.0, rmin_mm=80.0
        )
        # Start from a uniform intermediate density
        opt.x[:] = 0.7
        U, _, _ = opt.assemble_and_solve_static(opt.x, include_self_weight=False)
        dc = opt.compute_compliance_sensitivity(U, opt.x, include_self_weight=False)
        dc_f = opt.filter_sensitivity(dc, opt.x)
        x_new = opt.optimality_criteria_update(dc_f, move_limit=0.2)
        mean_new = float(np.mean(x_new))
        self.assertAlmostEqual(mean_new, volfrac, delta=0.05)
        self.assertTrue(np.all(np.isfinite(x_new)))
        # Compliance of the updated design must still be finite and positive
        U2, _, _ = opt.assemble_and_solve_static(x_new, include_self_weight=False)
        C2 = float(U2 @ domain.F)
        self.assertGreater(C2, 0.0)
        self.assertTrue(np.isfinite(C2))

    def test_self_weight_increases_compliance(self):
        domain = self._cantilever_domain()
        opt = TopologyOptimiser3DCompliance(
            domain, volfrac=1.0, penal_k=1.0, rmin_mm=80.0
        )
        opt.x[:] = 1.0
        U_no, _, _ = opt.assemble_and_solve_static(opt.x, include_self_weight=False)
        U_sw, _, _ = opt.assemble_and_solve_static(opt.x, include_self_weight=True)
        # Recover compliance consistently: C = U^T F_total
        F_sw = domain.F + opt.compute_self_weight_vector(opt.x)
        C_no = float(U_no @ domain.F)
        C_sw = float(U_sw @ F_sw)
        self.assertGreater(C_sw, C_no)
        self.assertTrue(np.isfinite(C_sw))


if __name__ == "__main__":
    unittest.main()
