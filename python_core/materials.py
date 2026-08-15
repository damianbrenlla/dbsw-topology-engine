"""
DBSW R260003 Topology Optimiser - Eurocode Material Registry
Elastic properties for structural steel (EC3), concrete (EC2), and timber (EC5).
"""

MATERIALS = {
    "EC3_STEEL": {
        "name": "Structural Steel (EC3)",
        "E0": 210e9,      # Young's Modulus (Pa)
        "nu": 0.30,       # Poisson's ratio
        "rho": 7850,      # Density (kg/m3)
        "fy": 355e6       # Characteristic yield strength (Pa)
    },
    "EC2_CONCRETE": {
        "name": "Concrete C30/37 (EC2)",
        "E0": 33e9,       # Secant modulus Ecm (Pa)
        "nu": 0.20,
        "rho": 2500,
        "fck": 30e6
    },
    "EC5_TIMBER": {
        "name": "Glulam GL28h (EC5)",
        "E0": 12.6e9,     # Mean modulus parallel to grain (Pa)
        "nu": 0.25,
        "rho": 430,
        "fmk": 28e6
    }
}


def get_material(mat_key: str) -> dict:
    return MATERIALS.get(mat_key, MATERIALS["EC3_STEEL"])
