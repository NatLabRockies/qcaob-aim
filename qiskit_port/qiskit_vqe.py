"""Qiskit VQE and variational Lanczos routines."""

import numpy as np


def expectation_value(state, Hmat):
    return np.vdot(state, Hmat @ state)


def transition_amplitude(left_state, Hmat, right_state):
    return np.vdot(left_state, Hmat @ right_state)


def overlap(left_state, right_state):
    return np.vdot(left_state, right_state)


def lanczos_cost_function_qiskit(
    params,
    Hmat,
    ut,
    b,
    i,
    ansatz_builder,
    n_layers,
    initial_occupations_indices,
    connected_graphs,
):
    """
    Qiskit version of vqe.py::lanczos_cost_function.

    params: current ansatz parameters
    Hmat: dense Hamiltonian matrix
    ut: list of previous Krylov statevectors as numpy arrays
    b: Lanczos b coefficients
    i: current Lanczos iteration index, starting at 1
    ansatz_builder: function that builds the Qiskit ansatz circuit
    """

    # Build trial Krylov state |w_i(theta)>
    circuit = ansatz_builder(
        params=params,
        n_layers=n_layers,
        initial_occupations_indices=initial_occupations_indices,
        connected_graphs=connected_graphs,
    )

    from qiskit.quantum_info import Statevector
    ws = Statevector.from_instruction(circuit).data

    # e1: enforce <w_i|H|u_{i-1}> = b_i
    e1 = abs(transition_amplitude(ws, Hmat, ut[i - 1]) - b[i])

    # e2/e3: enforce orthogonality with recent Krylov vectors
    e2 = abs(overlap(ws, ut[i - 1]))

    if i > 1:
        e3 = abs(overlap(ws, ut[i - 2]))
    else:
        e3 = 0.0

    # e4: extra orthogonality penalty for later Krylov vectors
    e4 = 0.0

    if i == 4:
        e4 += abs(overlap(ws, ut[1])) ** 2

    if i >= 5:
        for j in range(3, 6):
            k_overlap = abs(overlap(ws, ut[i - j])) ** 2
            e4 += k_overlap

    return e1**2 + e2**2 + e3**2 + e4


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


from openfermion import count_qubits
from openfermion.linalg import get_sparse_operator
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






