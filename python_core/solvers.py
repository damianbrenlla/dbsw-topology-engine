"""
DBSW R260003 Topology Optimiser - 3D SIMP Solver Engine
Sparse FE assembly, compliance minimisation, and Marching Cubes isosurface extraction.
"""

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from skimage.measure import marching_cubes
from .domain import Domain3D
from .materials import get_material


def hex8_stiffness_matrix(E: float, nu: float, dx: float, dy: float, dz: float) -> np.ndarray:
    """Precomputes 24x24 element stiffness matrix for a 3D Hex8 continuum element."""
    C = np.zeros((6, 6))
    factor = E / ((1 + nu) * (1 - 2 * nu))
    C[0, 0] = C[1, 1] = C[2, 2] = factor * (1 - nu)
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = factor * nu
    C[3, 3] = C[4, 4] = C[5, 5] = factor * (1 - 2 * nu) / 2.0

    gp = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    Ke = np.zeros((24, 24))

    for xi in gp:
        for eta in gp:
            for zeta in gp:
                dN_dxi = 0.125 * np.array([
                    [-(1-eta)*(1-zeta),  (1-eta)*(1-zeta),  (1+eta)*(1-zeta), -(1+eta)*(1-zeta),
                     -(1-eta)*(1+zeta),  (1-eta)*(1+zeta),  (1+eta)*(1+zeta), -(1+eta)*(1+zeta)],
                    [-(1-xi)*(1-zeta),  -(1+xi)*(1-zeta),   (1+xi)*(1-zeta),   (1-xi)*(1-zeta),
                     -(1-xi)*(1+zeta),  -(1+xi)*(1+zeta),   (1+xi)*(1+zeta),   (1-xi)*(1+zeta)],
                    [-(1-xi)*(1-eta),   -(1+xi)*(1-eta),   -(1+xi)*(1+eta),   -(1-xi)*(1+eta),
                      (1-xi)*(1-eta),    (1+xi)*(1-eta),    (1+xi)*(1+eta),    (1-xi)*(1+eta)]
                ])

                J = dN_dxi @ np.array([
                    [0, 0, 0], [dx, 0, 0], [dx, dy, 0], [0, dy, 0],
                    [0, 0, dz], [dx, 0, dz], [dx, dy, dz], [0, dy, dz]
                ])

                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                dN_dx = invJ @ dN_dxi

                B = np.zeros((6, 24))
                for i in range(8):
                    B[0, 3*i]   = dN_dx[0, i]
                    B[1, 3*i+1] = dN_dx[1, i]
                    B[2, 3*i+2] = dN_dx[2, i]
                    B[3, 3*i]   = dN_dx[1, i]; B[3, 3*i+1] = dN_dx[0, i]
                    B[4, 3*i+1] = dN_dx[2, i]; B[4, 3*i+2] = dN_dx[1, i]
                    B[5, 3*i]   = dN_dx[2, i]; B[5, 3*i+2] = dN_dx[0, i]

                Ke += B.T @ C @ B * detJ

    return Ke


def run_simp_optimisation(
    nelx: int, nely: int, nelz: int,
    lx: float, ly: float, lz: float,
    volfrac: float, max_iter: int,
    mat_key: str, penal: float = 3.0
):
    """Executes 3D SIMP topology optimization loop."""
    domain = Domain3D(nelx, nely, nelz, lx, ly, lz)
    mat = get_material(mat_key)

    Ke = hex8_stiffness_matrix(mat["E0"], mat["nu"], domain.dx, domain.dy, domain.dz)
    edof = domain.get_element_dofs()

    x = np.full((nelx, nely, nelz), volfrac)
    E_min = 1e-9 * mat["E0"]

    fixed_dofs = []
    for y in range(domain.nny):
        for z in range(domain.nnz):
            n1 = z * (domain.nnx * domain.nny) + y * domain.nnx + 0
            fixed_dofs.extend([3*n1, 3*n1+1, 3*n1+2])
            n2 = z * (domain.nnx * domain.nny) + y * domain.nnx + domain.nelx
            fixed_dofs.extend([3*n2+1, 3*n2+2])

    fixed_dofs = np.unique(fixed_dofs)
    free_dofs = np.setdiff1d(np.arange(domain.num_dofs), fixed_dofs)

    F = np.zeros(domain.num_dofs)
    center_x = domain.nelx // 2
    top_z = domain.nelz
    load_node = top_z * (domain.nnx * domain.nny) + (domain.nely // 2) * domain.nnx + center_x
    F[3 * load_node + 2] = -1000.0

    iK = np.kron(edof, np.ones((24, 1))).flatten()
    jK = np.kron(edof, np.ones((1, 24))).flatten()

    for it in range(max_iter):
        E_elem = E_min + (x.flatten() ** penal) * (mat["E0"] - E_min)
        sK = (Ke.flatten()[:, None] @ E_elem[None, :]).T.flatten()

        K = csc_matrix((sK, (iK, jK)), shape=(domain.num_dofs, domain.num_dofs))
        K_free = K[free_dofs, :][:, free_dofs]

        U_free = spsolve(K_free, F[free_dofs])
        U = np.zeros(domain.num_dofs)
        U[free_dofs] = U_free

        U_elem = U[edof]
        ce = np.sum((U_elem @ Ke) * U_elem, axis=1)
        c = np.sum((E_min + (x.flatten() ** penal) * (mat["E0"] - E_min)) * ce)

        dc = -penal * (x.flatten() ** (penal - 1)) * (mat["E0"] - E_min) * ce
        dc = dc.reshape((nelx, nely, nelz))

        l1, l2, move = 0.0, 1e9, 0.2
        while (l2 - l1) > 1e-4:
            lmid = 0.5 * (l2 + l1)
            x_candidate = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / lmid)))))
            if np.mean(x_candidate) - volfrac > 0:
                l1 = lmid
            else:
                l2 = lmid
            x = x_candidate

    verts, faces, normals, values = marching_cubes(x, level=0.5, spacing=(domain.dx, domain.dy, domain.dz))
    return verts.tolist(), faces.tolist(), float(c)
