"""
qiskit_port -- Qiskit port of the Anderson Impurity Model VQE/Green's-
function workflow.

Public surface only. Internal helpers (cost functions, ordering
conversions, sector-derivation logic, plotting internals) live in
qiskit_utils.py / qiskit_ansatz.py / qiskit_vqe.py / dmft_qiskit.py but
are intentionally NOT re-exported here -- import them from their
defining module directly if you need them.

Typical usage:

    from qiskit_port import run_aim_qiskit
    results = run_aim_qiskit(system_size=3, seed=0, vqe_depth=1)
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

from .dmft_qiskit import (
    initialize_system,
    compare_ground_states_qiskit,
    calculate_ground_state_qiskit,
    calculate_green_function_qiskit,
    calculate_relative_errors,
    run_aim_qiskit,
)

__all__ = [
    "create_qiskit_hamiltonian_matrix",
    "construct_vqe_gs_qiskit",
    "solve_vqe_qiskit",
    "vqe_krylov_zero_state_qiskit",
    "vqe_ideal_lanczos_iterations_qiskit",
    "vqe_gf_pm_qiskit",
    "calculate_gf_vqe_qiskit",
    "initialize_system",
    "compare_ground_states_qiskit",
    "calculate_ground_state_qiskit",
    "calculate_green_function_qiskit",
    "calculate_relative_errors",
    "run_aim_qiskit",
]
