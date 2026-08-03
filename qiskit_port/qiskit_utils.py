"""
Qiskit-side shared low-level helpers.

Base layer of the qiskit_port package -- these functions have no
dependency on the ansatz circuit, the optimizer, or any Green's-function
workflow. They are imported by qiskit_ansatz.py, qiskit_vqe.py, and
dmft_qiskit.py, and never import any of those files themselves (see the
project's dependency-direction rule: lower layers never import higher
ones).

Each function below is a direct, verified extraction from the original
qiskit_vqe.py -- same implementation, same behavior, just relocated so
it can be shared without pulling in the optimizer/Lanczos machinery
that used to live in the same file.
"""

import numpy as np
from openfermion import count_qubits
from openfermion.linalg import get_sparse_operator


def expectation_value(state, Hmat):
    return np.vdot(state, Hmat @ state)


def transition_amplitude(left_state, Hmat, right_state):
    return np.vdot(left_state, Hmat @ right_state)


def overlap(left_state, right_state):
    return np.vdot(left_state, right_state)


def continued_fraction_qiskit(w, a, b):
    assert len(a) == len(b)

    if len(a) == 1:
        return 1.0 / (w - a[0])

    return 1.0 / ((w - a[0]) - b[1] ** 2 * continued_fraction_qiskit(w, a[1:], b[1:]))


def qulacs_to_python_ordering_qiskit(qulacs_state, n_qubits):
    new_state = np.zeros(qulacs_state.shape, dtype=complex)

    for ii in range(qulacs_state.shape[0]):
        format_string = "{0:0" + str(n_qubits) + "b}"
        b_string = format_string.format(ii)
        b_reversed = b_string[::-1]
        new_ii = int(b_reversed, 2)
        new_state[new_ii] = qulacs_state[ii]

    return new_state


def create_qiskit_hamiltonian_matrix(qubit_hamiltonian, vqe_gs_energy):
    """
    Qiskit/NumPy equivalent of create_qulacs_hamiltonian().

    Returns:
    - shifted Hamiltonian matrix H
    - shifted squared Hamiltonian matrix H2
    - number of qubits
    """
    qubit_hamiltonian_shifted = qubit_hamiltonian - vqe_gs_energy

    n_qubits = count_qubits(qubit_hamiltonian_shifted)

    Hmat = get_sparse_operator(qubit_hamiltonian_shifted, n_qubits=n_qubits).toarray()
    H2mat = get_sparse_operator(qubit_hamiltonian_shifted ** 2, n_qubits=n_qubits).toarray()

    return Hmat, H2mat, n_qubits


def get_initial_occupations_indices_qiskit(
    up_qubit_indices,
    down_qubit_indices,
    n_up,
    n_down,
):
    most_equispaced_indices = np.linspace(
        0, len(up_qubit_indices) - 1, n_up, dtype=int
    ).tolist()
    initial_occupations_indices = [
        up_qubit_indices[i] for i in most_equispaced_indices
    ]

    most_equispaced_indices = np.linspace(
        0, len(down_qubit_indices) - 1, n_down, dtype=int
    ).tolist()
    initial_occupations_indices.extend(
        [down_qubit_indices[i] for i in most_equispaced_indices]
    )

    initial_occupations_indices.sort()
    return initial_occupations_indices
