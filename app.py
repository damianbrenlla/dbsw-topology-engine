# DBSW 3D Topology Web Server API Orchestrator (R260003 Topology Optimiser)
# Author: Damian Brenlla / DBSW 2026

import os
import sys
import time
import uuid
import threading
import webbrowser
import numpy as np
from flask import Flask, render_template, request, jsonify
from skimage import measure

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.domain import Domain3D
from core.materials import EurocodeMaterialRegistry
from core.solvers import TopologyOptimiser3DCompliance

app = Flask(__name__, template_folder="templates")
JOBS = {}

PENAL_INIT = 1.0
PENAL_FINAL = 3.0
PENAL_STEP = 0.5
NOTENSION_WEIGHT = 3.0


def open_browser():
    if os.environ.get("PORT") is None:
        webbrowser.open_new("http://127.0.0.1:5000/")


def async_optimise_worker(job_id, payload):
    try:
        t0 = time.time()
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["cancelled"] = False

        # 1. Resolve Material Properties via Eurocode Registry
        mat_props = EurocodeMaterialRegistry.resolve_properties(payload)

        # 2. Instantiate Spatial Domain3D
        domain = Domain3D(
            Lx=float(payload["Lx"]),
            Ly=float(payload["Ly"]),
            Lz=float(payload["Lz"]),
            nx=int(payload["nx"]),
            ny=int(payload["ny"]),
            nz=int(payload["nz"]),
            E=mat_props["E"],
            nu=mat_props["nu"],
            f_k=mat_props["f_k"],
            f_d=mat_props["f_d"],
            material_type=mat_props["material_type"],
            material_name=mat_props["material_name"],
            gamma_kn_m3=mat_props.get("gamma_kn_m3", 0.0),
        )

        # 3. Apply Boundary Support Restraints
        sup_mode = payload.get("support_mode", "preset")
        sup_preset = payload.get("support_preset", "cantilever")

        if sup_mode == "preset":
            if sup_preset == "cantilever":
                domain.add_support_box(
                    [0.0, 0.0], [0.0, domain.Ly], [0.0, domain.Lz], dofs="xyz"
                )
            elif sup_preset == "simply_supported":
                domain.add_support_box(
                    [0.0, 0.0], [0.0, domain.Ly], [0.0, 0.0], dofs="xyz"
                )
                domain.add_support_box(
                    [domain.Lx, domain.Lx],
                    [0.0, domain.Ly],
                    [0.0, 0.0],
                    dofs="yz",
                )
            elif sup_preset == "four_corners":
                domain.add_support_box([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], dofs="xyz")
                domain.add_support_box(
                    [domain.Lx, domain.Lx], [0.0, 0.0], [0.0, 0.0], dofs="xyz"
                )
                domain.add_support_box(
                    [0.0, 0.0], [domain.Ly, domain.Ly], [0.0, 0.0], dofs="xyz"
                )
                domain.add_support_box(
                    [domain.Lx, domain.Lx],
                    [domain.Ly, domain.Ly],
                    [0.0, 0.0],
                    dofs="xyz",
                )
        else:
            x_b = [
                float(payload.get("sup_x_min", 0)),
                float(payload.get("sup_x_max", 0)),
            ]
            y_b = [
                float(payload.get("sup_y_min", 0)),
                float(payload.get("sup_y_max", domain.Ly)),
            ]
            z_b = [
                float(payload.get("sup_z_min", 0)),
                float(payload.get("sup_z_max", domain.Lz)),
            ]
            domain.add_support_box(
                x_b, y_b, z_b, dofs=payload.get("sup_dofs", "xyz")
            )

        for ps in payload.get("point_supports", []):
            r = 15.0
            px, py, pz = float(ps["x"]), float(ps["y"]), float(ps["z"])
            domain.add_support_box(
                [px - r, px + r],
                [py - r, py + r],
                [pz - r, pz + r],
                dofs=ps.get("dofs", "xyz"),
            )

        # 4. Apply Loading Conditions
        load_preset = payload.get("load_preset", "custom")

        if load_preset == "top_udl":
            domain.add_patch_load(
                x_bounds=[0.0, domain.Lx],
                y_bounds=[0.0, domain.Ly],
                z_bounds=[domain.Lz - domain.dz, domain.Lz],
                total_load_xyz=[0.0, 0.0, -100000.0],
            )
        else:
            loads = payload.get("loads", [])
            if len(loads) == 0:
                domain.add_point_load(
                    [domain.Lx, domain.Ly / 2.0, domain.Lz], [0.0, 0.0, -100000.0]
                )
            else:
                for ld in loads:
                    px, py, pz = float(ld["x"]), float(ld["y"]), float(ld["z"])
                    fx = float(ld.get("Fx", 0.0)) * 1000.0
                    fy = float(ld.get("Fy", 0.0)) * 1000.0
                    fz = float(ld.get("Fz", -100.0)) * 1000.0
                    domain.add_point_load([px, py, pz], [fx, fy, fz])

        # 5. Instantiate SIMP Optimiser
        iterations = int(payload.get("iterations", 15))
        volfrac = float(payload.get("volfrac", 0.25))
        include_self_weight = bool(payload.get("include_self_weight", False))
        is_notension_material = domain.material_type in ("concrete", "masonry", "stone")

        opt = TopologyOptimiser3DCompliance(
            domain=domain,
            volfrac=volfrac,
            penal_k=PENAL_INIT,
            rmin_mm=150.0,
            notension_weight=NOTENSION_WEIGHT,
        )

        n_stages = int(round((PENAL_FINAL - PENAL_INIT) / PENAL_STEP)) + 1
        continuation_interval = max(1, iterations // n_stages)

        # 6. Compliance Optimisation Loop
        for i in range(iterations):
            if JOBS[job_id].get("cancelled", False):
                JOBS[job_id]["status"] = "cancelled"
                return

            if i > 0 and i % continuation_interval == 0:
                opt.penal_k = min(PENAL_FINAL, opt.penal_k + PENAL_STEP)

            JOBS[job_id]["current_iter"] = i + 1
            JOBS[job_id]["current_penal"] = round(opt.penal_k, 2)

            U, K_global, free_dofs = opt.assemble_and_solve_static(
                opt.x, include_self_weight=include_self_weight
            )

            U_elem = U[opt.edof_vec]
            element_energy_flat = np.sum((U_elem @ opt.KE) * U_elem, axis=1)
            element_energy = element_energy_flat.reshape(
                (domain.nx, domain.ny, domain.nz)
            )

            dc = (
                -opt.penal_k
                * (opt.x ** (opt.penal_k - 1.0))
                * (opt.Emax - opt.Emin)
                * element_energy
            )

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
                        np.minimum(
                            1.0,
                            np.minimum(
                                opt.x + move,
                                opt.x * np.sqrt(
                                    np.maximum(-dc_filtered, 1e-12) / lmid
                                ),
                            ),
                        ),
                    ),
                )
                xnew[domain.passive_mask == 1.0] = 1.0
                xnew[domain.passive_mask == 0.0] = 0.001

                if np.sum(xnew) - volfrac * opt.x.size > 0:
                    l1 = lmid
                else:
                    l2 = lmid
            opt.x = xnew

        # Final solve on final density field
        U_final, K_final, _ = opt.assemble_and_solve_static(
            opt.x, include_self_weight=include_self_weight
        )

        # 7. Surface Mesh Extraction & Field Recovery
        padded_x = np.pad(opt.x, 1, mode="constant", constant_values=0)

        # marching_cubes requires the iso-level to sit strictly between the data's
        # min and max. At low volume fractions (or with too few OC iterations at a
        # given continuation stage) the density field can legitimately fail to
        # cross the nominal 0.5 threshold anywhere in the domain, which used to
        # raise an unhandled ValueError and surface as a generic "Solver Error".
        # Fall back to an adaptive level based on the density range actually
        # achieved, so a valid (if more conservative) surface is always extracted.
        data_min = float(padded_x.min())
        data_max = float(padded_x.max())

        # A fixed level=0.50 assumes the SIMP field is nearly binarised (0/1)
        # everywhere. That's a reasonable assumption for high-volfrac designs,
        # but at low volume fractions the optimiser genuinely doesn't need much
        # density in low-demand regions (e.g. near a pinned support, where
        # bending moment is near zero for a simply supported UDL beam) even
        # after full convergence. Those regions stay "grey" (intermediate
        # density) rather than snapping to 0 or 1. Extracting at a fixed 0.50
        # then silently drops that grey material from the mesh, which reads as
        # a disconnected/floating structure even though the underlying field
        # is a single continuous load path. Scaling the level to the target
        # volume fraction keeps the extracted geometry consistent with what
        # the optimiser actually decided to keep.
        SOLID_LEVEL = min(0.50, max(0.05, volfrac))

        if data_max <= data_min:
            raise ValueError(
                "Optimisation produced a uniform density field with no load path — "
                "try increasing iterations, raising the volume fraction, or checking "
                "that loads/supports are connected."
            )

        if SOLID_LEVEL <= data_min or SOLID_LEVEL >= data_max:
            level = data_min + 0.5 * (data_max - data_min)
            JOBS[job_id]["level_note"] = (
                f"Density field did not fully binarise (max density {data_max:.3f}); "
                f"surface extracted at adaptive threshold {level:.3f} instead of 0.50. "
                f"Consider more iterations or a higher volume fraction for a crisper result."
            )
        else:
            level = SOLID_LEVEL

        verts, faces, _, _ = measure.marching_cubes(padded_x, level=level)

        # Recover Signed Cauchy stresses (MPa)
        elem_stresses_mpa = opt.recover_element_stress_field(U_final)
        stress_grid_3d = elem_stresses_mpa.reshape(
            (domain.nx, domain.ny, domain.nz)
        )
        padded_stress = np.pad(stress_grid_3d, 1, mode="edge")

        # Recover nodal displacements (mm)
        U_nodes, disp_mags = opt.recover_nodal_displacements(U_final)
        disp_grid_3d = disp_mags.reshape(
            (domain.nx + 1, domain.ny + 1, domain.nz + 1)
        )
        padded_disp = np.pad(disp_grid_3d, 1, mode="edge")

        # Sample Signed Stress & Displacement Fields at Mesh Surface Vertices
        vertex_stresses_mpa = []
        vertex_deflections_mm = []

        for v in verts:
            ix = int(np.clip(round(v[0]), 0, domain.nx + 1))
            iy = int(np.clip(round(v[1]), 0, domain.ny + 1))
            iz = int(np.clip(round(v[2]), 0, domain.nz + 1))

            vertex_stresses_mpa.append(float(padded_stress[ix, iy, iz]))
            vertex_deflections_mm.append(float(padded_disp[ix, iy, iz]))

        # Convert verts from padded grid indices to physical mm coordinates
        verts_mm = np.copy(verts)
        verts_mm[:, 0] = (verts[:, 0] - 1.0) * domain.dx
        verts_mm[:, 1] = (verts[:, 1] - 1.0) * domain.dy
        verts_mm[:, 2] = (verts[:, 2] - 1.0) * domain.dz

        # Max Structural Limits
        x_flat = opt.x.flatten()
        solid_elem_stresses = elem_stresses_mpa[x_flat > 0.1]

        if len(solid_elem_stresses) > 0:
            sigma_max_tens = float(np.max(solid_elem_stresses))
            sigma_max_comp = float(np.min(solid_elem_stresses))
        else:
            sigma_max_tens = float(np.max(vertex_stresses_mpa))
            sigma_max_comp = float(np.min(vertex_stresses_mpa))

        sigma_max_abs = max(abs(sigma_max_tens), abs(sigma_max_comp))
        u_max = float(np.max(disp_mags))

        elapsed_time = time.time() - t0

        terminal_keys = [
            k for k, v in JOBS.items()
            if v.get("status") in ["completed", "error", "cancelled"] and k != job_id
        ]
        if len(JOBS) > 20 and terminal_keys:
            JOBS.pop(terminal_keys[0])

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["solve_time_s"] = float(np.round(elapsed_time, 2))
        JOBS[job_id]["vertices"] = verts_mm.tolist()
        JOBS[job_id]["faces"] = faces.tolist()
        JOBS[job_id]["stresses_mpa"] = vertex_stresses_mpa
        JOBS[job_id]["deflections_mm"] = vertex_deflections_mm
        JOBS[job_id]["sigma_max_abs"] = sigma_max_abs
        JOBS[job_id]["sigma_max_tens"] = sigma_max_tens
        JOBS[job_id]["sigma_max_comp"] = sigma_max_comp
        JOBS[job_id]["u_max"] = u_max
        JOBS[job_id]["f_k"] = mat_props["f_k"]
        JOBS[job_id]["f_d"] = mat_props["f_d"]
        JOBS[job_id]["material_name"] = mat_props["material_name"]

    except Exception as e:
        print(f"[Solver Error in Job {job_id}]: {e}")
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["message"] = str(e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start_optimise", methods=["POST"])
def start_optimise():
    payload = request.get_json()
    job_id = str(uuid.uuid4())[:8]

    JOBS[job_id] = {
        "status": "queued",
        "current_iter": 0,
        "total_iter": int(payload.get("iterations", 15)),
        "cancelled": False,
    }

    thread = threading.Thread(
        target=async_optimise_worker, args=(job_id, payload)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "job_id": job_id})


@app.route("/api/stop_optimise/<job_id>", methods=["POST"])
def stop_optimise(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"status": "error", "message": "Job ID not found"}), 404

    job["cancelled"] = True
    job["status"] = "cancelled"
    return jsonify({"status": "stopping", "job_id": job_id})


@app.route("/api/status/<job_id>", methods=["GET"])
def get_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"status": "error", "message": "Job ID not found"}), 404
    return jsonify(job)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    is_cloud = os.environ.get("PORT") is not None
    bind_ip = "0.0.0.0" if is_cloud else "127.0.0.1"

    print("\n" + "=" * 68)
    print(" DBSW STRUCTURAL ENGINE — 3D TOPOLOGY WEB SERVER")
    print(f" Port Binding: {bind_ip}:{port} | Environment: {'Cloud' if is_cloud else 'Local'}")
    print("=" * 68 + "\n")

    if not is_cloud:
        threading.Timer(1.2, open_browser).start()

    app.run(host=bind_ip, port=port, debug=False)