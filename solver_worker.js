// DBSW R260003 Topology Engine — Client-Side Pyodide/Wasm Web Worker
// Author: Damian Brenlla / DBSW 2026

importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js");

let pyodide = null;

/**
 * Boots the WebAssembly CPython runtime and pre-loads scientific packages
 * directly into the client's browser local memory.
 */
async function initPyodideEngine() {
    try {
        postMessage({ status: "log", message: "Initialising WebAssembly Python runtime..." });
        pyodide = await loadPyodide();

        postMessage({ status: "log", message: "Loading NumPy, SciPy & Scikit-Image into browser..." });
        await pyodide.loadPackage(["numpy", "scipy", "scikit-image"]);

        postMessage({ status: "log", message: "Importing DBSW Python structural core into Wasm VFS..." });

        // Copy Python source files into Pyodide's Virtual File System (VFS)
        const files = ["domain.py", "materials.py", "solvers.py", "__init__.py"];
        for (const file of files) {
            const response = await fetch(`./python_core/${file}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch ./python_core/${file} (HTTP ${response.status})`);
            }
            const code = await response.text();
            pyodide.FS.writeFile(`/home/pyodide/core/${file}`, code);
        }

        // Configure Pyodide sys.path and verify core imports
        await pyodide.runPythonAsync(`
import sys
import os
sys.path.append('/home/pyodide')

from core.domain import Domain3D
from core.materials import EurocodeMaterialRegistry
from core.solvers import TopologyOptimiser3DCompliance
from skimage import measure
from scipy.ndimage import label
import numpy as np
import json
        `);

        postMessage({ status: "ready", message: "DBSW Pyodide Engine Ready." });
    } catch (err) {
        postMessage({ status: "error", message: `Wasm Engine Init Failed: ${err.message}` });
    }
}

/**
 * Listens for solve requests dispatched from index.html (Three.js UI)
 */
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
            postMessage({ status: "running", current_iter: 1, total_iter: payload.iterations });

            // Pass payload to Python environment
            pyodide.globals.set("payload_json", JSON.stringify(payload));

            // Execute compliance optimization loop inside WebAssembly
            const resultJson = await pyodide.runPythonAsync(`
payload = json.loads(payload_json)

# 1. Material Resolution
mat_props = EurocodeMaterialRegistry.resolve_properties(payload)

# 2. Domain Initialization
domain = Domain3D(
    Lx=float(payload["Lx"]), Ly=float(payload["Ly"]), Lz=float(payload["Lz"]),
    nx=int(payload["nx"]), ny=int(payload["ny"]), nz=int(payload["nz"]),
    E=mat_props["E"], nu=mat_props["nu"], f_k=mat_props["f_k"], f_d=mat_props["f_d"],
    material_type=mat_props["material_type"], material_name=mat_props["material_name"]
)

# 3. Boundary Restraints
sup_mode = payload.get("support_mode", "preset")
if sup_mode == "preset":
    domain.add_support_box([0.0, 0.0], [0.0, domain.Ly], [0.0, 0.0], dofs="xyz")
    domain.add_support_box([domain.Lx, domain.Lx], [0.0, domain.Ly], [0.0, 0.0], dofs="yz")

# 4. Loading Definitions
for ld in payload.get("loads", []):
    domain.add_point_load(
        [float(ld["x"]), float(ld["y"]), float(ld["z"])], 
        [float(ld["Fx"])*1000.0, float(ld["Fy"])*1000.0, float(ld["Fz"])*1000.0]
    )

# 5. SIMP Optimiser Initialization
volfrac = float(payload.get("volfrac", 0.20))
iterations = int(payload.get("iterations", 30))
opt = TopologyOptimiser3DCompliance(domain=domain, volfrac=volfrac, penal_k=1.0, rmin_mm=150.0)

# 6. Optimization Loop
for i in range(iterations):
    if i > 0 and i % max(1, iterations // 5) == 0:
        opt.penal_k = min(3.0, opt.penal_k + 0.5)
    
    U, K, free = opt.assemble_and_solve_static(opt.x)
    U_elem = U[opt.edof_vec]
    element_energy = np.sum((U_elem @ opt.KE) * U_elem, axis=1).reshape((domain.nx, domain.ny, domain.nz))
    dc = -opt.penal_k * (opt.x ** (opt.penal_k - 1.0)) * (opt.Emax - opt.Emin) * element_energy
    dc_filtered = opt.filter_sensitivity(dc, opt.x)

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
        if np.sum(xnew) - volfrac * opt.x.size > 0:
            l1 = lmid
        else:
            l2 = lmid
    opt.x = xnew

# 7. Final Solved State & Connected Component Isosurface
U_final, _, _ = opt.assemble_and_solve_static(opt.x)
padded_x = np.pad(opt.x, 1, mode="constant", constant_values=0)
SOLID_LEVEL = max(0.35, min(0.50, volfrac * 1.5))

binary_grid = padded_x >= SOLID_LEVEL
labeled_grid, num_features = label(binary_grid)

if num_features > 1:
    component_sizes = np.bincount(labeled_grid.ravel())
    component_sizes[0] = 0  # Ignore void background
    main_component_id = np.argmax(component_sizes)
    padded_x[labeled_grid != main_component_id] = 0.0

data_max = float(padded_x.max())
level = SOLID_LEVEL if data_max >= SOLID_LEVEL else max(0.10, data_max * 0.8)

verts, faces, _, _ = measure.marching_cubes(padded_x, level=level)

# 8. Physical mm Coordinate Mapping & Field Sampling
verts_mm = np.copy(verts)
verts_mm[:, 0] = (verts[:, 0] - 1.0) * domain.dx
verts_mm[:, 1] = (verts[:, 1] - 1.0) * domain.dy
verts_mm[:, 2] = (verts[:, 2] - 1.0) * domain.dz

elem_stresses = opt.recover_element_stress_field(U_final)
padded_stress = np.pad(elem_stresses.reshape((domain.nx, domain.ny, domain.nz)), 1, mode="edge")

vertex_stresses = []
for v in verts:
    ix = int(np.clip(round(v[0]), 0, domain.nx + 1))
    iy = int(np.clip(round(v[1]), 0, domain.ny + 1))
    iz = int(np.clip(round(v[2]), 0, domain.nz + 1))
    vertex_stresses.append(float(padded_stress[ix, iy, iz]))

json.dumps({
    "vertices": verts_mm.tolist(),
    "faces": faces.tolist(),
    "stresses_mpa": vertex_stresses,
    "sigma_max_abs": float(np.max(np.abs(elem_stresses))),
    "u_max": float(np.max(opt.recover_nodal_displacements(U_final)[1]))
})
            `);

            postMessage({ status: "completed", data: JSON.parse(resultJson) });
        } catch (err) {
            postMessage({ status: "error", message: err.toString() });
        }
    }
};