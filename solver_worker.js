importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js");

let pyodide = null;

async function initEngine() {
    try {
        self.postMessage({ type: "STATUS", message: "Downloading Pyodide core engine..." });
        pyodide = await loadPyodide();

        self.postMessage({ type: "STATUS", message: "Loading NumPy & SciPy sparse matrix solver (Wasm)..." });
        await pyodide.loadPackage(["numpy", "scipy"]);

        self.postMessage({ type: "STATUS", message: "Loading Scikit-Image Marching Cubes module..." });
        await pyodide.loadPackage("scikit-image");

        self.postMessage({ type: "STATUS", message: "Fetching Eurocode Python FEA modules..." });
        const files = ["domain.py", "materials.py", "solvers.py"];
        const modules = {};

        for (const file of files) {
            const response = await fetch(`./python_core/${file}?v=${Date.now()}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status} fetching python_core/${file}`);
            }
            modules[file] = await response.text();
        }

        self.postMessage({ type: "STATUS", message: "Injecting module registry into Pyodide RAM..." });
        await pyodide.runPythonAsync(`
import sys
import types

python_core = types.ModuleType("python_core")
sys.modules["python_core"] = python_core

domain_mod = types.ModuleType("python_core.domain")
exec("""${modules["domain.py"].replace(/\\/g, "\\\\").replace(/`/g, "\\`").replace(/\$/g, "\\$")}""", domain_mod.__dict__)
sys.modules["python_core.domain"] = domain_mod

mat_mod = types.ModuleType("python_core.materials")
exec("""${modules["materials.py"].replace(/\\/g, "\\\\").replace(/`/g, "\\`").replace(/\$/g, "\\$")}""", mat_mod.__dict__)
sys.modules["python_core.materials"] = mat_mod

solv_mod = types.ModuleType("python_core.solvers")
exec("""${modules["solvers.py"].replace(/\\/g, "\\\\").replace(/`/g, "\\`").replace(/\$/g, "\\$")}""", solv_mod.__dict__)
sys.modules["python_core.solvers"] = solv_mod
        `);

        self.postMessage({ type: "READY" });
    } catch (err) {
        self.postMessage({ type: "ERROR", message: err.message });
    }
}

self.onmessage = async function (e) {
    if (e.data.type === "INIT") {
        await initEngine();
    } else if (e.data.type === "RUN") {
        try {
            const params = e.data.params;
            const pyCode = `
import json
import python_core.solvers as solvers

verts, faces, compliance = solvers.run_simp_optimisation(
    nelx=${parseInt(params.nelx)},
    nely=${parseInt(params.nely)},
    nelz=${parseInt(params.nelz)},
    lx=${parseFloat(params.lx)},
    ly=${parseFloat(params.ly)},
    lz=${parseFloat(params.lz)},
    volfrac=${parseFloat(params.volfrac)},
    max_iter=${parseInt(params.max_iter)},
    mat_key="${params.mat_key}"
)

json.dumps({"verts": verts, "faces": faces, "compliance": compliance})
            `;
            const resultJson = await pyodide.runPythonAsync(pyCode);
            const result = JSON.parse(resultJson);
            self.postMessage({ type: "SUCCESS", data: result });
        } catch (err) {
            self.postMessage({ type: "ERROR", message: `Solver Execution Error: ${err.message}` });
        }
    }
};
