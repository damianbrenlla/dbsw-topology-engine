importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js");

let pyodide = null;

async function initPyodideEngine() {
    try {
        pyodide = await loadPyodide();
        
        // Load heavy numerical C-extensions into browser Wasm memory
        await pyodide.loadPackage(["numpy", "scipy", "scikit-image"]);

        // Fetch Python source files over HTTP from GitHub Pages CDN
        const files = ["domain.py", "materials.py", "solvers.py"];
        const modules = {};

        for (const file of files) {
            const response = await fetch(`./python_core/${file}?v=${Date.now()}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status} fetching python_core/${file}`);
            }
            modules[file] = await response.text();
        }

        // Create python_core module in Pyodide's native module registry
        await pyodide.runPythonAsync(`
import sys
import types

# Create in-memory package space
python_core = types.ModuleType("python_core")
sys.modules["python_core"] = python_core

# Execute domain module
domain_mod = types.ModuleType("python_core.domain")
exec("""${modules["domain.py"].replace(/\\/g, "\\\\").replace(/`/g, "\\`").replace(/\$/g, "\\$")}""", domain_mod.__dict__)
sys.modules["python_core.domain"] = domain_mod
python_core.domain = domain_mod

# Execute materials module
mat_mod = types.ModuleType("python_core.materials")
exec("""${modules["materials.py"].replace(/\\/g, "\\\\").replace(/`/g, "\\`").replace(/\$/g, "\\$")}""", mat_mod.__dict__)
sys.modules["python_core.materials"] = mat_mod
python_core.materials = mat_mod

# Execute solvers module
solv_mod = types.ModuleType("python_core.solvers")
exec("""${modules["solvers.py"].replace(/\\/g, "\\\\").replace(/`/g, "\\`").replace(/\$/g, "\\$")}""", solv_mod.__dict__)
sys.modules["python_core.solvers"] = solv_mod
python_core.solvers = solv_mod
        `);

        self.postMessage({ type: "READY" });
    } catch (err) {
        self.postMessage({ type: "ERROR", message: `Wasm Init Error: ${err.message}` });
    }
}

self.onmessage = async function (e) {
    const data = e.data;
    if (data.type === "INIT") {
        await initPyodideEngine();
    } else if (data.type === "RUN") {
        try {
            const pyCode = `
import json
import python_core.solvers as solvers

verts, faces, compliance = solvers.run_simp_optimisation(
    nelx=${parseInt(data.params.nelx)},
    nely=${parseInt(data.params.nely)},
    nelz=${parseInt(data.params.nelz)},
    lx=${parseFloat(data.params.lx)},
    ly=${parseFloat(data.params.ly)},
    lz=${parseFloat(data.params.lz)},
    volfrac=${parseFloat(data.params.volfrac)},
    max_iter=${parseInt(data.params.max_iter)},
    mat_key="${data.params.mat_key}"
)

json.dumps({"verts": verts, "faces": faces, "compliance": compliance})
            `;
            
            const resultJson = await pyodide.runPythonAsync(pyCode);
            const result = JSON.parse(resultJson);
            self.postMessage({ type: "SUCCESS", data: result });
        } catch (err) {
            self.postMessage({ type: "ERROR", message: `Solver Error: ${err.message}` });
        }
    }
};
