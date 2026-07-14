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




from qiskit.quantum_info import Statevector

from qiskit_port.qiskit_ansatz import all_symmetry_ansatzae_qiskit


def construct_vqe_gs_qiskit(
    n_qubits,
    up_qubit_indices,
    down_qubit_indices,
    vqe_depth,
    connected_graphs,
    minimum_angles,
    vqe_nu,
    vqe_nd,
):
    """
    Reconstruct a variational state from previously determined ansatz angles.

    Qiskit equivalent of vqe.construct_vqe_gs().
    """

    initial_occupations_indices = get_initial_occupations_indices_qiskit(
        up_qubit_indices,
        down_qubit_indices,
        vqe_nu,
        vqe_nd,
    )

    circuit = all_symmetry_ansatzae_qiskit(
        theta=minimum_angles,
        n_qubits=n_qubits,
        n_layers=vqe_depth,
        initial_occupations_indices=initial_occupations_indices,
        connected_graphs=connected_graphs,
        compilation="generic",
    )

    statevector = Statevector.from_instruction(circuit)
    state_array = statevector.data.copy()

    return statevector, state_array

import numpy as np
from qiskit.quantum_info import Statevector


def _apply_jordan_wigner_project_flip(
    state,
    impurity_orbital,
    occupied_projector,
):
    """
    Apply the Jordan-Wigner parity string, project onto the requested
    impurity occupation, and flip the impurity qubit.

    occupied_projector=True corresponds to P1 followed by X.
    occupied_projector=False corresponds to P0 followed by X.
    """
    state = np.asarray(state, dtype=complex)
    output = np.zeros_like(state)

    for basis_index, amplitude in enumerate(state):
        impurity_bit = (basis_index >> impurity_orbital) & 1

        required_bit = 1 if occupied_projector else 0

        # P1 or P0 projection
        if impurity_bit != required_bit:
            continue

        # Jordan-Wigner Z string on qubits below the impurity orbital
        parity = 1.0

        for qubit in range(impurity_orbital):
            bit = (basis_index >> qubit) & 1

            if bit == 1:
                parity *= -1.0

        # X gate flips the impurity qubit
        flipped_index = basis_index ^ (1 << impurity_orbital)

        output[flipped_index] += parity * amplitude

    norm = np.linalg.norm(output)

    if norm == 0:
        raise ValueError(
            "The projected Krylov zero state has zero norm."
        )

    output /= norm

    return output, norm


def vqe_krylov_zero_state_qiskit(
    vqe_gs,
    impurity_orbital,
    n_qubits,
):
    """
    Qiskit equivalent of vqe.vqe_krylov_zero_state().

    Creates normalized particle-removal and particle-addition
    Krylov zero states from the reconstructed VQE state.
    """
    if isinstance(vqe_gs, Statevector):
        ground_state = vqe_gs.data
    else:
        ground_state = np.asarray(vqe_gs, dtype=complex)

    expected_dimension = 2**n_qubits

    if ground_state.size != expected_dimension:
        raise ValueError(
            f"Expected a statevector of length {expected_dimension}, "
            f"but received length {ground_state.size}."
        )

    # Phi minus: Jordan-Wigner parity string, P1, then X
    phi_minus_array, phi_minus_norm = (
        _apply_jordan_wigner_project_flip(
            state=ground_state,
            impurity_orbital=impurity_orbital,
            occupied_projector=True,
        )
    )

    # Phi plus: Jordan-Wigner parity string, P0, then X
    phi_plus_array, phi_plus_norm = (
        _apply_jordan_wigner_project_flip(
            state=ground_state,
            impurity_orbital=impurity_orbital,
            occupied_projector=False,
        )
    )

    phi_minus = Statevector(phi_minus_array)
    phi_plus = Statevector(phi_plus_array)

    record_keeping = {
        "phi_plus_norm": phi_plus_norm,
        "phi_plus_normalized": np.linalg.norm(phi_plus_array),
        "phi_minus_norm": phi_minus_norm,
        "phi_minus_normalized": np.linalg.norm(phi_minus_array),
    }

    return (
        phi_minus,
        phi_minus_array,
        phi_minus_norm,
        phi_plus,
        phi_plus_array,
        phi_plus_norm,
        record_keeping,
    )


import numpy as np
from qiskit.quantum_info import Statevector

from qiskit_port.qiskit_ansatz import all_symmetry_ansatzae_qiskit


def lanczos_cost_function_qiskit(
    params,
    Hmat,
    ut,
    b,
    i,
    n_layers,
    initial_occupations_indices,
    connected_graphs,
    n_qubits,
    compilation="generic",
):
    """
    Qiskit equivalent of vqe.lanczos_cost_function().
    """

    circuit = all_symmetry_ansatzae_qiskit(
        theta=params,
        n_qubits=n_qubits,
        n_layers=n_layers,
        initial_occupations_indices=initial_occupations_indices,
        connected_graphs=connected_graphs,
        compilation=compilation,
    )

    # Raw Qiskit statevector
    ws_raw = Statevector.from_instruction(circuit).data

    # OpenFermion matrices require the reversed bit ordering
    ws_ordered = qulacs_to_python_ordering_qiskit(
        ws_raw,
        n_qubits,
    )

    previous_ordered = qulacs_to_python_ordering_qiskit(
        ut[i - 1],
        n_qubits,
    )

    # Enforce <ws|H|u_(i-1)> = b_i
    e1 = abs(
        np.vdot(
            ws_ordered,
            Hmat @ previous_ordered,
        )
        - b[i]
    )

    # Overlaps can be calculated directly in raw statevector ordering
    e2 = abs(np.vdot(ws_raw, ut[i - 1]))

    if i > 1:
        e3 = abs(np.vdot(ws_raw, ut[i - 2]))
    else:
        e3 = 0.0

    e4 = 0.0

    if i == 4:
        e4 += abs(np.vdot(ws_raw, ut[1])) ** 2

    if i >= 5:
        for j in range(3, 6):
            e4 += abs(np.vdot(ws_raw, ut[i - j])) ** 2

    cost = e1**2 + e2**2 + e3**2 + e4

    return float(np.real(cost))


from scipy.optimize import minimize


def vqe_ideal_lanczos_iterations_qiskit(
    niter,
    u,
    Hmat,
    H2mat,
    charge_sec,
    spin_sec,
    n_layers,
    up_qubit_indices,
    down_qubit_indices,
    connected_graphs,
    conv_tol=1e-6,
    optimizer="COBYLA",
    maxiter=2000,
    gf_gtol=5e-5,
):
    """
    Qiskit equivalent of vqe.vqe_ideal_lanczos_iterations().

    Parameters
    ----------
    u
        Initial Krylov statevector in raw Qiskit/Qulacs ordering.
    Hmat, H2mat
        Shifted Hamiltonian matrices in OpenFermion matrix ordering.
    """

    u = np.asarray(
        u.data if isinstance(u, Statevector) else u,
        dtype=complex,
    )

    n_qubits = int(np.log2(u.size))
    n_edges = connected_graphs["graph_full"].number_of_edges()

    n_up = (charge_sec + spin_sec) // 2
    n_down = (charge_sec - spin_sec) // 2

    initial_occupations_indices = (
        get_initial_occupations_indices_qiskit(
            up_qubit_indices,
            down_qubit_indices,
            n_up,
            n_down,
        )
    )

    compilation = "generic"

    a = np.zeros(niter + 1, dtype=np.complex128)
    b = np.zeros(niter + 1, dtype=np.complex128)

    ut = [u.copy()]
    b[0] = 0.0

    # Hmat uses reordered state indexing
    u0_ordered = qulacs_to_python_ordering_qiskit(
        ut[0],
        n_qubits,
    )

    a[0] = np.vdot(
        u0_ordered,
        Hmat @ u0_ordered,
    )

    iteration_results = {}

    for i in range(1, niter + 1):
        previous_ordered = qulacs_to_python_ordering_qiskit(
            ut[i - 1],
            n_qubits,
        )

        h2_expectation = np.vdot(
            previous_ordered,
            H2mat @ previous_ordered,
        )

        if i == 1:
            b_squared = (
                h2_expectation
                - a[i - 1] ** 2
            )
        else:
            b_squared = (
                h2_expectation
                - a[i - 1] ** 2
                - b[i - 1] ** 2
            )

        # Remove tiny negative numerical noise
        if abs(b_squared.imag) < 1e-12:
            b_squared = b_squared.real

        if np.isreal(b_squared) and b_squared < 0:
            if abs(b_squared) < 1e-10:
                b_squared = 0.0
            else:
                raise ValueError(
                    f"Negative b² encountered at iteration {i}: "
                    f"{b_squared}"
                )

        b[i] = np.sqrt(b_squared)

        if np.isclose(b[i], 0.0):
            break

        n_params = n_layers * (n_edges + n_qubits)

        theta_0 = np.random.uniform(
            low=0.0,
            high=2.0 * np.pi,
            size=n_params,
        )

        opt_args = (
            Hmat,
            ut,
            b,
            i,
            n_layers,
            initial_occupations_indices,
            connected_graphs,
            n_qubits,
            compilation,
        )

        result = minimize(
            lanczos_cost_function_qiskit,
            theta_0,
            args=opt_args,
            method=optimizer,
            tol=conv_tol,
            options={
                "maxiter": int(maxiter),
                "gtol": gf_gtol,
                "eps": 1e-7,
            },
        )

        circuit = all_symmetry_ansatzae_qiskit(
            theta=result.x,
            n_qubits=n_qubits,
            n_layers=n_layers,
            initial_occupations_indices=(
                initial_occupations_indices
            ),
            connected_graphs=connected_graphs,
            compilation=compilation,
        )

        new_state = Statevector.from_instruction(circuit).data
        new_state = new_state / np.linalg.norm(new_state)

        ut.append(new_state.copy())

        new_state_ordered = (
            qulacs_to_python_ordering_qiskit(
                new_state,
                n_qubits,
            )
        )

        a[i] = np.vdot(
            new_state_ordered,
            Hmat @ new_state_ordered,
        )

        iteration_results[i] = {
            "message": result.message,
            "success": result.success,
            "status": result.status,
            "fun": result.fun,
            "parameters": result.x,
        }

    return a, b, iteration_results, ut