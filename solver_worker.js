/**
 * DBSW R260004 Topology Engine - Client-Side Pyodide WebWorker
 * Author: Damian Brenlla / DBSW 2026
 *
 * This mirrors the orchestration logic in the Flask app.py version (support
 * presets/custom boxes, point supports, line supports, load presets/point loads,
 * optional self-weight, approximate no-tension sensitivity penalty, penal_k
 * continuation) but runs it inside Pyodide instead of a server process.
 *
 * The optimisation loop runs ONE ITERATION PER `await pyodide.runPythonAsync`
 * call so the worker yields control back to the JS event loop after every
 * iteration and can postMessage live progress.
 *
 * FEATURES & FIXES:
 * 1. Resampled nodal displacement field onto element-centered grid via 8-node
 *    corner averaging prior to padding to match padded_stress shape.
 * 2. Applied exact half-voxel origin alignment and explicit coordinate clamping
 *    to [0, Lx], [0, Ly], [0, Lz] for flush Three.js mesh alignment.
 * 3. v2 — FIXED 3D Line Support Discretisation. The previous approach sampled
 *    points along P1->P2 and called domain.add_support_box() with a small
 *    isotropic search-box radius at each sample. add_support_box() has no
 *    real node-finding tolerance of its own (eps=1e-2mm, meant only to
 *    absorb floating-point rounding) -- so capture depended entirely on that
 *    worker-side radius being large enough on EVERY axis. On an anisotropic
 *    mesh (e.g. the default Lx=6000 Ly=300 Lz=600, nx=36 ny=9 nz=12 -> dx
 *    166.7mm, dy 33.3mm, dz 50mm), a radius sized off the mesh's finest
 *    spacing (dy) is too small on the coarser axes (dx). Confirmed by direct
 *    test: a line support placed more than ~12.5mm from an existing node row
 *    on Y -- a ~21mm-wide gap out of every 33.3mm between rows -- captured
 *    ZERO degrees of freedom, silently. Fixed by delegating to a new
 *    domain.add_line_support(p1, p2, dofs) method (see domain.py) that snaps
 *    each sample to its nearest node index per axis instead of testing
 *    box containment -- the same proven approach add_point_load() already
 *    uses. (Measured: for a single boundary line support, the new call is
 *    about as fast as the old one -- this fix is about correctness, not
 *    performance, since the old box-sampling's early per-axis rejection
 *    was already fairly cheap for the boundary-aligned lines that happened
 *    to be the only cases tested before.)
 */

importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js");

let pyodide = null;

// Defaults (overridden by payload when provided)
const DEFAULT_PENAL_INIT = 1.0;
const DEFAULT_PENAL_FINAL = 3.0;
const DEFAULT_PENAL_STEP = 0.5;
const DEFAULT_NOTENSION_WEIGHT = 3.0;
const DEFAULT_RMIN_MM = 150.0;
const DEFAULT_MOVE_LIMIT = 0.2;

// Kept in worker memory between messages so the client can request intermediate frames
let densityHistory = null; // list of density grids (Python side) or null
let lastDomainMeta = null; // {Lx,Ly,Lz,nx,ny,nz,dx,dy,dz} for frame extraction

async function initPyodideEngine() {
    try {
        postMessage({ status: "log", message: "Initialising WebAssembly Python runtime..." });
        pyodide = await loadPyodide();

        postMessage({ status: "log", message: "Loading NumPy, SciPy & Scikit-Image into browser memory..." });
        await pyodide.loadPackage(["numpy", "scipy", "scikit-image"]);

        postMessage({ status: "log", message: "Mounting DBSW Python structural core into Wasm MEMFS..." });

        pyodide.FS.mkdirTree("/home/pyodide/core");

        const files = ["domain.py", "materials.py", "solvers.py", "__init__.py"];
        for (const file of files) {
            const response = await fetch(`./python_core/${file}?cb=${Date.now()}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch ./python_core/${file} (HTTP ${response.status})`);
            }
            const code = await response.text();
            pyodide.FS.writeFile(`/home/pyodide/core/${file}`, code);
        }

        await pyodide.runPythonAsync(`
import sys
if '/home/pyodide' not in sys.path:
    sys.path.append('/home/pyodide')

from core.domain import Domain3D
from core.materials import EurocodeMaterialRegistry
from core.solvers import TopologyOptimiser3DCompliance
from skimage import measure
from scipy.ndimage import label
import numpy as np
import json
import time
        `);

        postMessage({ status: "ready", message: "DBSW Pyodide Engine Ready." });
    } catch (err) {
        postMessage({ status: "error", message: `Wasm Engine Init Failed: ${err.message}` });
    }
}

self.onmessage = async function(e) {
    const { action, payload } = e.data;

    if (action === "init") {
        await initPyodideEngine();
        return;
    }

    if (action === "solve") {
        if (!pyodide) {
            postMessage({ status: "error", message: "Pyodide engine is not initialized yet." });
            return;
        }

        try {
            const iterations = Number.parseInt(payload.iterations, 10);
            if (!Number.isInteger(iterations) || iterations < 1 || iterations > 500) {
                throw new Error("Iterations must be an integer between 1 and 500.");
            }

            // Advanced SIMP parameters (with safe defaults)
            const penalInit = Number.isFinite(parseFloat(payload.penal_init)) ? parseFloat(payload.penal_init) : DEFAULT_PENAL_INIT;
            const penalFinal = Number.isFinite(parseFloat(payload.penal_final)) ? parseFloat(payload.penal_final) : DEFAULT_PENAL_FINAL;
            const penalStep = Number.isFinite(parseFloat(payload.penal_step)) ? Math.max(0.1, parseFloat(payload.penal_step)) : DEFAULT_PENAL_STEP;
            const rminMm = Number.isFinite(parseFloat(payload.rmin_mm)) ? Math.max(1.0, parseFloat(payload.rmin_mm)) : DEFAULT_RMIN_MM;
            const moveLimit = Number.isFinite(parseFloat(payload.move_limit)) ? Math.max(0.01, Math.min(0.5, parseFloat(payload.move_limit))) : DEFAULT_MOVE_LIMIT;
            const notensionWeight = Number.isFinite(parseFloat(payload.notension_weight)) ? Math.max(0, parseFloat(payload.notension_weight)) : DEFAULT_NOTENSION_WEIGHT;
            const recordHistory = payload.record_density_history !== false; // default on
            if (!Number.isFinite(penalInit) || !Number.isFinite(penalFinal) || penalInit < 1 || penalFinal < penalInit || penalFinal > 5) {
                throw new Error("Penalisation must satisfy 1 ≤ start ≤ final ≤ 5.");
            }
            if (!Number.isFinite(penalStep) || penalStep <= 0) {
                throw new Error("Penalisation step must be a positive finite number.");
            }

            postMessage({ status: "running", current_iter: 0, total_iter: iterations, phase: "Resolving material & domain" });

            // Clear previous history
            densityHistory = null;
            lastDomainMeta = null;

            pyodide.globals.set("payload_json", JSON.stringify(payload));
            pyodide.globals.set("PENAL_INIT", penalInit);
            pyodide.globals.set("PENAL_FINAL", penalFinal);
            pyodide.globals.set("PENAL_STEP", penalStep);
            pyodide.globals.set("RMIN_MM", rminMm);
            pyodide.globals.set("MOVE_LIMIT", moveLimit);
            pyodide.globals.set("NOTENSION_WEIGHT", notensionWeight);
            pyodide.globals.set("RECORD_HISTORY", recordHistory);

            // 1. ONE-OFF SETUP: material, domain, supports, line supports, loads, optimiser instance.
            await pyodide.runPythonAsync(`
t0 = time.time()
payload = json.loads(payload_json)

mat_props = EurocodeMaterialRegistry.resolve_properties(payload)

domain = Domain3D(
    Lx=float(payload["Lx"]), Ly=float(payload["Ly"]), Lz=float(payload["Lz"]),
    nx=int(payload["nx"]), ny=int(payload["ny"]), nz=int(payload["nz"]),
    E=mat_props["E"], nu=mat_props["nu"], f_k=mat_props["f_k"], f_d=mat_props["f_d"],
    material_type=mat_props["material_type"], material_name=mat_props["material_name"],
    gamma_kn_m3=mat_props.get("gamma_kn_m3", 0.0),
)

# --- Void / Forced-Solid Passive Regions ---
void_regions = payload.get("void_regions", [])
if void_regions:
    _rx = (np.arange(domain.nx) + 0.5) * domain.dx
    _ry = (np.arange(domain.ny) + 0.5) * domain.dy
    _rz = (np.arange(domain.nz) + 0.5) * domain.dz
    _RX, _RY, _RZ = np.meshgrid(_rx, _ry, _rz, indexing="ij")
    for region in void_regions:
        _region_mask = (
            (_RX >= float(region["x_min"])) & (_RX <= float(region["x_max"])) &
            (_RY >= float(region["y_min"])) & (_RY <= float(region["y_max"])) &
            (_RZ >= float(region["z_min"])) & (_RZ <= float(region["z_max"]))
        )
        domain.passive_mask[_region_mask] = 1.0 if region.get("region_type", "void") == "solid" else 0.0

# --- Boundary Support Restraints ---
sup_mode = payload.get("support_mode", "preset")
sup_preset = payload.get("support_preset", "cantilever")

if sup_mode == "preset":
    if sup_preset == "cantilever":
        domain.add_support_box([0.0, 0.0], [0.0, domain.Ly], [0.0, domain.Lz], dofs="xyz")
    elif sup_preset == "simply_supported":
        domain.add_support_box([0.0, 0.0], [0.0, domain.Ly], [0.0, 0.0], dofs="xyz")
        domain.add_support_box([domain.Lx, domain.Lx], [0.0, domain.Ly], [0.0, 0.0], dofs="yz")
    elif sup_preset == "four_corners":
        domain.add_support_box([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], dofs="xyz")
        domain.add_support_box([domain.Lx, domain.Lx], [0.0, 0.0], [0.0, 0.0], dofs="xyz")
        domain.add_support_box([0.0, 0.0], [domain.Ly, domain.Ly], [0.0, 0.0], dofs="xyz")
        domain.add_support_box([domain.Lx, domain.Lx], [domain.Ly, domain.Ly], [0.0, 0.0], dofs="xyz")
    elif sup_preset == "all_edges":
        edge_t = max(domain.dx, domain.dy, domain.dz) * 1.5
        domain.add_support_box([0.0, domain.Lx], [0.0, edge_t], [0.0, 0.0], dofs="xyz")
        domain.add_support_box([0.0, domain.Lx], [domain.Ly - edge_t, domain.Ly], [0.0, 0.0], dofs="xyz")
        domain.add_support_box([0.0, edge_t], [0.0, domain.Ly], [0.0, 0.0], dofs="xyz")
        domain.add_support_box([domain.Lx - edge_t, domain.Lx], [0.0, domain.Ly], [0.0, 0.0], dofs="xyz")
elif sup_mode == "custom":
    x_b = [float(payload.get("sup_x_min", 0)), float(payload.get("sup_x_max", 0))]
    y_b = [float(payload.get("sup_y_min", 0)), float(payload.get("sup_y_max", domain.Ly))]
    z_b = [float(payload.get("sup_z_min", 0)), float(payload.get("sup_z_max", domain.Lz))]
    domain.add_support_box(x_b, y_b, z_b, dofs=payload.get("sup_dofs", "xyz"))

# Discrete Point Supports — snap to the nearest mesh node on each axis.
for ps in payload.get("point_supports", []):
    coord = [float(ps["x"]), float(ps["y"]), float(ps["z"])]
    domain.add_point_support(coord, dofs=ps.get("dofs", "xyz"))

# --- Discrete Line Supports Discretisation ---
# FIX: delegate to domain.add_line_support(), which snaps each sampled
# point along the line to its nearest node index per axis (dx, dy, dz
# handled independently) instead of testing containment inside a small
# search box. The old box-based approach here silently captured zero
# nodes -- and therefore restrained nothing -- whenever a line support
# was placed more than a few mm off an axis with coarse mesh spacing.
# See domain.py's add_line_support() docstring and this file's header
# comment for the confirmed failure case and reproduction numbers.
line_supports = payload.get("line_supports", [])
for ls in line_supports:
    p1 = [float(ls["x1"]), float(ls["y1"]), float(ls["z1"])]
    p2 = [float(ls["x2"]), float(ls["y2"]), float(ls["z2"])]
    domain.add_line_support(p1, p2, dofs=ls.get("dofs", "xyz"))

# --- Loading Conditions ---
load_preset = payload.get("load_preset", "custom")

if load_preset == "top_udl":
    domain.add_patch_load(
        x_bounds=[0.0, domain.Lx], y_bounds=[0.0, domain.Ly],
        z_bounds=[domain.Lz - domain.dz, domain.Lz],
        total_load_xyz=[0.0, 0.0, -100000.0],
    )
else:
    for ld in payload.get("loads", []):
        px, py, pz = float(ld["x"]), float(ld["y"]), float(ld["z"])
        fx = float(ld.get("Fx", 0.0)) * 1000.0
        fy = float(ld.get("Fy", 0.0)) * 1000.0
        fz = float(ld.get("Fz", -100.0)) * 1000.0
        domain.add_point_load([px, py, pz], [fx, fy, fz])

# --- Optimiser Setup ---
volfrac = float(payload.get("volfrac", 0.20))
sim_iterations = int(payload.get("iterations", 15))
if not np.isfinite(volfrac) or not (0.0 < volfrac <= 1.0):
    raise ValueError("Volume fraction must be greater than 0 and no greater than 1.")
include_self_weight = payload.get("include_self_weight", True) is True
is_notension_material = domain.material_type in ("concrete", "masonry", "stone")

opt = TopologyOptimiser3DCompliance(
    domain=domain, volfrac=volfrac, penal_k=PENAL_INIT, rmin_mm=RMIN_MM,
    notension_weight=NOTENSION_WEIGHT,
)

passive_solid = domain.passive_mask == 1.0
passive_void = domain.passive_mask == 0.0
free_elements = ~(passive_solid | passive_void)
target_volume = volfrac * domain.n_elems
fixed_volume = float(np.count_nonzero(passive_solid)) + 0.001 * float(np.count_nonzero(passive_void))
n_free = int(np.count_nonzero(free_elements))
if fixed_volume + 0.001 * n_free > target_volume + 1e-8:
    raise ValueError("Requested volume fraction is infeasible for the selected passive solid/void regions.")
initial_free_density = 0.001 if n_free == 0 else min(1.0, max(0.001, (target_volume - fixed_volume) / n_free))
opt.x = np.full((domain.nx, domain.ny, domain.nz), initial_free_density)
opt.x[passive_solid] = 1.0
opt.x[passive_void] = 0.001

# Penalisation is advanced continuously through the requested stage increments.
# This preserves the configured step size as the total continuation schedule,
# but removes abrupt stiffness-model jumps at fixed iteration boundaries.

# Density history for evolution scrubber (list of float32 arrays)
density_history = [] if RECORD_HISTORY else None
if density_history is not None:
    density_history.append(np.round(opt.x.astype(np.float32), 3).copy())
            `);

            // 2. ONE SIMP ITERATION PER AWAIT
            for (let i = 0; i < iterations; i++) {
                pyodide.globals.set("i", i);
                const t0 = performance.now();

                await pyodide.runPythonAsync(`
opt.penal_k = opt.continuation_penalisation(
    i, sim_iterations, PENAL_INIT, PENAL_FINAL, PENAL_STEP
)

U, K_global, free_dofs = opt.assemble_and_solve_static(opt.x, include_self_weight=include_self_weight)

dc = opt.compute_compliance_sensitivity(
    U, opt.x, include_self_weight=include_self_weight
)
dc_filtered = opt.filter_sensitivity(dc, opt.x)

# The no-tension term is retained as an explicitly heuristic material
# preference; it is not a consistent derivative of the compliance objective.
if is_notension_material and NOTENSION_WEIGHT > 0.0:
    dc_filtered = opt.combine_notension_sensitivity(
        dc_filtered, opt.compute_notension_penalty(U), opt.x
    )

opt.x = opt.optimality_criteria_update(dc_filtered, move_limit=MOVE_LIMIT)

if density_history is not None:
    density_history.append(np.round(opt.x.astype(np.float32), 3).copy())
                `);

                const elapsed_ms = Math.round(performance.now() - t0);
                postMessage({
                    status: "running", current_iter: i + 1, total_iter: iterations,
                    last_iter_ms: elapsed_ms, phase: "Solving"
                });
            }

            postMessage({ status: "running", current_iter: iterations, total_iter: iterations, phase: "Extracting mesh & recovering fields" });

            // 3. Final solve on x_final + mesh extraction + stress/displacement field recovery.
            const resultJson = await pyodide.runPythonAsync(`
U_final, K_final, _ = opt.assemble_and_solve_static(opt.x, include_self_weight=include_self_weight)

padded_x = np.pad(opt.x, 1, mode="constant", constant_values=0)
verts, faces, _, _ = measure.marching_cubes(padded_x, level=0.50)

# Element stress field (nx, ny, nz)
elem_stresses_mpa = opt.recover_element_stress_field(U_final)
stress_grid_3d = elem_stresses_mpa.reshape((domain.nx, domain.ny, domain.nz))
padded_stress = np.pad(stress_grid_3d, 1, mode="edge")

# Nodal displacement field (nx+1, ny+1, nz+1)
U_nodes, disp_mags = opt.recover_nodal_displacements(U_final)
disp_grid_3d = disp_mags.reshape((domain.nx + 1, domain.ny + 1, domain.nz + 1))

# Resample node-centered displacement grid onto element-centered grid via 8-node corner averaging
nx, ny, nz = domain.nx, domain.ny, domain.nz
disp_elem_3d = 0.125 * (
    disp_grid_3d[0:nx,   0:ny,   0:nz]   +
    disp_grid_3d[1:nx+1, 0:ny,   0:nz]   +
    disp_grid_3d[0:nx,   1:ny+1, 0:nz]   +
    disp_grid_3d[1:nx+1, 1:ny+1, 0:nz]   +
    disp_grid_3d[0:nx,   0:ny,   1:nz+1] +
    disp_grid_3d[1:nx+1, 0:ny,   1:nz+1] +
    disp_grid_3d[0:nx,   1:ny+1, 1:nz+1] +
    disp_grid_3d[1:nx+1, 1:ny+1, 1:nz+1]
)
padded_disp = np.pad(disp_elem_3d, 1, mode="edge")

vertex_stresses_mpa = []
vertex_deflections_mm = []
for v in verts:
    ix = int(np.clip(round(v[0]), 0, domain.nx + 1))
    iy = int(np.clip(round(v[1]), 0, domain.ny + 1))
    iz = int(np.clip(round(v[2]), 0, domain.nz + 1))
    vertex_stresses_mpa.append(float(padded_stress[ix, iy, iz]))
    vertex_deflections_mm.append(float(padded_disp[ix, iy, iz]))

verts_mm = np.copy(verts)
verts_mm[:, 0] = (verts[:, 0] - 1.0) * domain.dx
verts_mm[:, 1] = (verts[:, 1] - 1.0) * domain.dy
verts_mm[:, 2] = (verts[:, 2] - 1.0) * domain.dz

verts_mm[:, 0] = np.clip(verts_mm[:, 0], 0.0, domain.Lx)
verts_mm[:, 1] = np.clip(verts_mm[:, 1], 0.0, domain.Ly)
verts_mm[:, 2] = np.clip(verts_mm[:, 2], 0.0, domain.Lz)

x_flat = opt.x.flatten()
solid_elem_stresses = elem_stresses_mpa[x_flat > 0.1]
if len(solid_elem_stresses) > 0:
    sigma_max_tens = float(np.max(solid_elem_stresses))
    sigma_max_comp = float(np.min(solid_elem_stresses))
else:
    sigma_max_tens = float(np.max(vertex_stresses_mpa))
    sigma_max_comp = float(np.min(vertex_stresses_mpa))

sigma_max_abs = max(abs(sigma_max_tens), abs(sigma_max_comp))

if len(vertex_deflections_mm) > 0:
    u_max = float(np.max(np.abs(vertex_deflections_mm)))
else:
    u_max = float(np.max(disp_mags))

# Reaction recovery (residual forces at restrained DOFs, grouped by support)
reactions = opt.compute_reactions(U_final, K_final, include_self_weight=include_self_weight)

elapsed_time = time.time() - t0

n_history = len(density_history) if density_history is not None else 0

json.dumps({
    "vertices": verts_mm.tolist(),
    "faces": faces.tolist(),
    "stresses_mpa": vertex_stresses_mpa,
    "deflections_mm": vertex_deflections_mm,
    "sigma_max_abs": sigma_max_abs,
    "sigma_max_tens": sigma_max_tens,
    "sigma_max_comp": sigma_max_comp,
    "u_max": u_max,
    "f_k": mat_props["f_k"],
    "f_d": mat_props["f_d"],
    "material_name": mat_props["material_name"],
    "solve_time_s": round(elapsed_time, 2),
    "self_weight_included": include_self_weight,
    "notension_applied": is_notension_material,
    "notension_method": "heuristic sensitivity preference (not a constitutive no-tension material model)",
    "filter_radius_requested_mm": opt.requested_rmin_mm,
    "filter_radius_applied_mm": opt.rmin_mm,
    "density_history_frames": n_history,
    "reactions": reactions,
})
            `);

            // Keep density history alive in Python globals for on-demand extract_frame
            if (recordHistory) {
                await pyodide.runPythonAsync(`
# Persist history + domain metrics for later frame extraction
_density_history = density_history
_domain_dx = float(domain.dx)
_domain_dy = float(domain.dy)
_domain_dz = float(domain.dz)
_domain_Lx = float(domain.Lx)
_domain_Ly = float(domain.Ly)
_domain_Lz = float(domain.Lz)
_domain_nx = int(domain.nx)
_domain_ny = int(domain.ny)
_domain_nz = int(domain.nz)
                `);
                densityHistory = true; // flag that history is available
            } else {
                densityHistory = null;
            }

            postMessage({ status: "completed", data: JSON.parse(resultJson) });
        } catch (err) {
            postMessage({ status: "error", message: err.toString() });
        }
        return;
    }

    // --- On-demand isosurface extraction for density evolution scrubber ---
    if (action === "extract_frame") {
        if (!pyodide || !densityHistory) {
            postMessage({ status: "error", message: "No density history available. Re-run with 'Record density history' enabled." });
            return;
        }
        try {
            const frameIndex = parseInt(payload.frame_index) || 0;
            pyodide.globals.set("frame_index", frameIndex);
            const frameJson = await pyodide.runPythonAsync(`
idx = int(frame_index)
if _density_history is None or idx < 0 or idx >= len(_density_history):
    raise ValueError(f"Frame index {idx} out of range (0..{len(_density_history)-1 if _density_history else -1})")

x_frame = _density_history[idx]
padded = np.pad(x_frame, 1, mode="constant", constant_values=0)
verts, faces, _, _ = measure.marching_cubes(padded, level=0.50)

verts_mm = np.copy(verts)
verts_mm[:, 0] = np.clip((verts[:, 0] - 1.0) * _domain_dx, 0.0, _domain_Lx)
verts_mm[:, 1] = np.clip((verts[:, 1] - 1.0) * _domain_dy, 0.0, _domain_Ly)
verts_mm[:, 2] = np.clip((verts[:, 2] - 1.0) * _domain_dz, 0.0, _domain_Lz)

json.dumps({
    "vertices": verts_mm.tolist(),
    "faces": faces.tolist(),
    "frame_index": idx,
    "total_frames": len(_density_history),
})
            `);
            postMessage({ status: "frame_extracted", data: JSON.parse(frameJson) });
        } catch (err) {
            postMessage({ status: "error", message: "Frame extraction failed: " + err.toString() });
        }
        return;
    }
};
