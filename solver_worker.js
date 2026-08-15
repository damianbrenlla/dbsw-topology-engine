importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js");

let pyodide = null;

async function initEngine() {
    try {
        self.postMessage({ type: "STATUS", message: "Downloading Pyodide core engine..." });
        pyodide = await loadPyodide();

        self.postMessage({ type: "STATUS", message: "Loading NumPy & SciPy (Wasm)..." });
        await pyodide.loadPackage(["numpy", "scipy"]);

        self.postMessage({ type: "STATUS", message: "Loading Scikit-Image..." });
        await pyodide.loadPackage("scikit-image");

        self.postMessage({ type: "STATUS", message: "Fetching Eurocode Python FEA modules..." });

        // Ensure Virtual Filesystem directory structure exists
        try { pyodide.FS.mkdir('/python_core'); } catch (e) { /* directory already exists */ }

        const files = ["__init__.py", "domain.py", "materials.py", "solvers.py"];

        for (const file of files) {
            const response = await fetch(`./python_core/${file}?v=${Date.now()}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status} fetching python_core/${file}`);
            }
            const content = await response.text();
            // Write raw string directly into Pyodide's virtual RAM filesystem
            pyodide.FS.writeFile(`/python_core/${file}`, content);
        }

        self.postMessage({ type: "STATUS", message: "Mounting Python modules into sys.path..." });
        await pyodide.runPythonAsync(`
import sys
if '/' not in sys.path:
    sys.path.append('/')

import python_core.solvers as solvers
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
