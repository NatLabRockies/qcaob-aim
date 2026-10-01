"""
Qiskit backend support for the Anderson Impurity Model workflow.

The high-level AIM/DMFT workflow now lives in the shared dmft.py driver.
Select the quantum backend there with:

    BACKEND = "qiskit"

This package contains only Qiskit-specific lower-level utilities,
ansatz construction, VQE routines, and variational-Lanczos routines.
"""

from .qiskit_utils import (
    create_qiskit_hamiltonian_matrix,
)

from .qiskit_vqe import (
    construct_vqe_gs_qiskit,
    solve_vqe_qiskit,
    vqe_krylov_zero_state_qiskit,
    vqe_ideal_lanczos_iterations_qiskit,
    vqe_gf_pm_qiskit,
    calculate_gf_vqe_qiskit,
)

__all__ = [
    "create_qiskit_hamiltonian_matrix",
    "construct_vqe_gs_qiskit",
    "solve_vqe_qiskit",
    "vqe_krylov_zero_state_qiskit",
    "vqe_ideal_lanczos_iterations_qiskit",
    "vqe_gf_pm_qiskit",
    "calculate_gf_vqe_qiskit",
]
