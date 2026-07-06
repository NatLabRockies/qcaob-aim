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














    