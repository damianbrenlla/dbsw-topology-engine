/**
 * DBSW R260003 Topology Engine - Client-Side Pyodide WebWorker
 * Author: Damian Brenlla / DBSW 2026
 *
 * Soft-void + robust nodal displacement sampling (2026-10 fix)
 * - Hard void only in the deep interior of keep-out boxes
 * - Thin transition band left free so the density filter can create a smooth ramp
 * - Displacement colouring uses true continuous nodal field with trilinear interpolation
 */
importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js");

let pyodide = null;
const PENAL_INIT = 1.0;
const PENAL_FINAL = 3.0;
const PENAL_STEP = 0.5;
const NOTENSION_WEIGHT = 3.0;

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
            const iterations = parseInt(payload.iterations) || 15;
            postMessage({ status: "running", current_iter: 0, total_iter: iterations, phase: "Resolving material & domain" });
            pyodide.globals.set("payload_json", JSON.stringify(payload));
            pyodide.globals.set("PENAL_INIT", PENAL_INIT);
            pyodide.globals.set("PENAL_FINAL", PENAL_FINAL);
            pyodide.globals.set("PENAL_STEP", PENAL_STEP);
            pyodide.globals.set("NOTENSION_WEIGHT", NOTENSION_WEIGHT);

            // 1. ONE-OFF SETUP
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

# --- Soft Void / Forced-Solid Passive Regions ---
void_regions = payload.get("void_regions", [])
if void_regions:
    _rx = (np.arange(domain.nx) + 0.5) * domain.dx
    _ry = (np.arange(domain.ny) + 0.5) * domain.dy
    _rz = (np.arange(domain.nz) + 0.5) * domain.dz
    _RX, _RY, _RZ = np.meshgrid(_rx, _ry, _rz, indexing="ij")

    # Transition width ≈ 1.5 × largest element size
    trans = 1.5 * max(domain.dx, domain.dy, domain.dz)

    for region in void_regions:
        x0 = float(region["x_min"])
        x1 = float(region["x_max"])
        y0 = float(region["y_min"])
        y1 = float(region["y_max"])
        z0 = float(region["z_min"])
        z1 = float(region["z_max"])
        is_solid = region.get("region_type", "void") == "solid"

        if is_solid:
            # Forced solid stays hard
            mask = (
                (_RX >= x0) & (_RX <= x1) &
                (_RY >= y0) & (_RY <= y1) &
                (_RZ >= z0) & (_RZ <= z1)
            )
            domain.passive_mask[mask] = 1.0
        else:
            # Soft void: hard only deep inside, transition band left free
            dist = np.maximum.reduce([
                np.maximum(x0 - _RX, 0.0),
                np.maximum(_RX - x1, 0.0),
                np.maximum(y0 - _RY, 0.0),
                np.maximum(_RY - y1, 0.0),
                np.maximum(z0 - _RZ, 0.0),
                np.maximum(_RZ - z1, 0.0)
            ])
            inside = (dist == 0.0)
            domain.passive_mask[inside] = 0.0          # hard void core
            # near region (0 < dist < trans) stays -1 (free) → filter creates smooth ramp

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

# Discrete Point Supports
for ps in payload.get("point_supports", []):
    r = 15.0
    px, py, pz = float(ps["x"]), float(ps["y"]), float(ps["z"])
    domain.add_support_box([px - r, px + r], [py - r, py + r], [pz - r, pz + r], dofs=ps.get("dofs", "xyz"))

# Discrete Line Supports
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
include_self_weight = bool(payload.get("include_self_weight", True))
is_notension_material = domain.material_type in ("concrete", "masonry", "stone")
opt = TopologyOptimiser3DCompliance(
    domain=domain, volfrac=volfrac, penal_k=PENAL_INIT, rmin_mm=150.0,
    notension_weight=NOTENSION_WEIGHT,
)
opt.x = np.full((domain.nx, domain.ny, domain.nz), volfrac)
opt.x[domain.passive_mask == 1.0] = 1.0
opt.x[domain.passive_mask == 0.0] = 0.001
n_stages = int(round((PENAL_FINAL - PENAL_INIT) / PENAL_STEP)) + 1
continuation_interval = max(1, sim_iterations // n_stages)
            `);

            // 2. ONE SIMP ITERATION PER AWAIT
            for (let i = 0; i < iterations; i++) {
                pyodide.globals.set("i", i);
                const t0 = performance.now();
                await pyodide.runPythonAsync(`
if i > 0 and i % continuation_interval == 0:
    opt.penal_k = min(PENAL_FINAL, opt.penal_k + PENAL_STEP)
U, K_global, free_dofs = opt.assemble_and_solve_static(opt.x, include_self_weight=include_self_weight)
U_elem = U[opt.edof_vec]
element_energy = np.sum((U_elem @ opt.KE) * U_elem, axis=1).reshape((domain.nx, domain.ny, domain.nz))
dc = -opt.penal_k * (opt.x ** (opt.penal_k - 1.0)) * (opt.Emax - opt.Emin) * element_energy
dc_filtered = opt.filter_sensitivity(dc, opt.x)
if is_notension_material:
    dc_filtered = dc_filtered + opt.compute_notension_penalty(U)
l1, l2, move = 0.0, 1e12, 0.2
while (l2 - l1) / (l1 + l2 + 1e-10) > 1e-4:
    lmid = 0.5 * (l1 + l2)
    xnew = np.maximum(
        0.001,
        np.maximum(
            opt.x - move,
            np.minimum(1.0, np.minimum(opt.x + move, opt.x * np.sqrt(np.maximum(-dc_filtered, 1e-12) / lmid)))
        )
    )
    xnew[domain.passive_mask == 1.0] = 1.0
    xnew[domain.passive_mask == 0.0] = 0.001   # only hard void core
    if np.sum(xnew) - volfrac * opt.x.size > 0:
        l1 = lmid
    else:
        l2 = lmid
opt.x = xnew
                `);
                const elapsed_ms = Math.round(performance.now() - t0);
                postMessage({
                    status: "running", current_iter: i + 1, total_iter: iterations,
                    last_iter_ms: elapsed_ms, phase: "Solving"
                });
            }

            postMessage({ status: "running", current_iter: iterations, total_iter: iterations, phase: "Extracting mesh & recovering fields" });

            // 3. Final solve + mesh + robust nodal displacement recovery
            const resultJson = await pyodide.runPythonAsync(`
U_final, K_final, _ = opt.assemble_and_solve_static(opt.x, include_self_weight=include_self_weight)
padded_x = np.pad(opt.x, 1, mode="constant", constant_values=0)
verts, faces, _, _ = measure.marching_cubes(padded_x, level=0.50)

# Element stress field (unchanged)
elem_stresses_mpa = opt.recover_element_stress_field(U_final)
stress_grid_3d = elem_stresses_mpa.reshape((domain.nx, domain.ny, domain.nz))
padded_stress = np.pad(stress_grid_3d, 1, mode="edge")

# --- Robust nodal displacement field ---
U_nodes, disp_mags = opt.recover_nodal_displacements(U_final)
# disp_mags shape = (nx+1, ny+1, nz+1)

vertex_stresses_mpa = []
vertex_deflections_mm = []

for v in verts:                       # v is in padded element-index space
    # Stress (still element-based)
    ix = int(np.clip(round(v[0]), 0, domain.nx + 1))
    iy = int(np.clip(round(v[1]), 0, domain.ny + 1))
    iz = int(np.clip(round(v[2]), 0, domain.nz + 1))
    vertex_stresses_mpa.append(float(padded_stress[ix, iy, iz]))

    # Displacement – true continuous nodal field with trilinear interpolation
    fx = np.clip(v[0] - 1.0, 0.0, domain.nx)   # remove padding offset
    fy = np.clip(v[1] - 1.0, 0.0, domain.ny)
    fz = np.clip(v[2] - 1.0, 0.0, domain.nz)

    ix0 = int(np.floor(fx)); ix1 = min(ix0 + 1, domain.nx)
    iy0 = int(np.floor(fy)); iy1 = min(iy0 + 1, domain.ny)
    iz0 = int(np.floor(fz)); iz1 = min(iz0 + 1, domain.nz)

    wx = fx - ix0
    wy = fy - iy0
    wz = fz - iz0

    c000 = disp_mags[ix0, iy0, iz0]
    c100 = disp_mags[ix1, iy0, iz0]
    c010 = disp_mags[ix0, iy1, iz0]
    c110 = disp_mags[ix1, iy1, iz0]
    c001 = disp_mags[ix0, iy0, iz1]
    c101 = disp_mags[ix1, iy0, iz1]
    c011 = disp_mags[ix0, iy1, iz1]
    c111 = disp_mags[ix1, iy1, iz1]

    c00 = c000 * (1.0 - wx) + c100 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx
    c0  = c00  * (1.0 - wy) + c10  * wy
    c1  = c01  * (1.0 - wy) + c11  * wy
    val = c0   * (1.0 - wz) + c1   * wz

    vertex_deflections_mm.append(float(val))

# Convert vertices to mm and clamp
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
    sigma_max_tens = float(np.max(vertex_stresses_mpa)) if vertex_stresses_mpa else 0.0
    sigma_max_comp = float(np.min(vertex_stresses_mpa)) if vertex_stresses_mpa else 0.0
sigma_max_abs = max(abs(sigma_max_tens), abs(sigma_max_comp))

u_max = float(np.max(np.abs(vertex_deflections_mm))) if vertex_deflections_mm else float(np.max(disp_mags))

elapsed_time = time.time() - t0
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
})
            `);
            postMessage({ status: "completed", data: JSON.parse(resultJson) });
        } catch (err) {
            postMessage({ status: "error", message: err.toString() });
        }
    }
};
