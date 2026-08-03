"""
Qiskit VQE and variational Lanczos routines.

Second layer of the qiskit_port package: ground-state reconstruction,
Krylov-state construction, the Lanczos cost function and iteration loop,
the Green's-function branch calculator, and the ground-state VQE
optimizer with its full charge/spin sector search.

Imports the ansatz circuit builder from qiskit_ansatz.py and the shared
low-level helpers from qiskit_utils.py -- this file does NOT redefine
expectation_value / overlap / transition_amplitude / ordering conversion
/ Hamiltonian-matrix construction / initial-occupation selection
locally anymore; those live in qiskit_utils.py now and are imported
from there, so there is exactly one implementation of each, not two
that can silently drift apart.
"""

import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import Statevector

from qiskit_port.qiskit_ansatz import all_symmetry_ansatzae_qiskit
from qiskit_port.qiskit_utils import (
    expectation_value,
    overlap,
    transition_amplitude,
    continued_fraction_qiskit,
    qulacs_to_python_ordering_qiskit,
    create_qiskit_hamiltonian_matrix,
    get_initial_occupations_indices_qiskit,
)
from n_site_graph_creation import create_connected_graphs, AIMSiteModelsEnum


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
    initial_thetas=None,
):
    """
    Qiskit equivalent of vqe.vqe_ideal_lanczos_iterations().

    This version mirrors the original Qulacs implementation while using
    Qiskit statevectors and NumPy/OpenFermion Hamiltonian matrices.

    Parameters
    ----------
    niter : int
        Number of variational Lanczos iterations.

    u : array-like or Statevector
        Initial Krylov vector in raw Qiskit/Qulacs circuit ordering.

    Hmat : np.ndarray
        Shifted Hamiltonian matrix.

    H2mat : np.ndarray
        Squared shifted Hamiltonian matrix.

    charge_sec : int
        Charge sector.

    spin_sec : int
        Spin sector.

    n_layers : int
        Number of ansatz layers.

    up_qubit_indices : list[int]
        Spin-up qubit indices.

    down_qubit_indices : list[int]
        Spin-down qubit indices.

    connected_graphs : dict
        AIM connected graphs.

    conv_tol : float
        SciPy optimizer convergence tolerance.

    optimizer : str
        SciPy optimizer.

    maxiter : int
        Maximum optimizer function-evaluation budget.

    gf_gtol : float
        Retained for compatibility with the original function interface.

    initial_thetas : list[np.ndarray] or None
        Optional fixed starting vectors.

        initial_thetas[0] -> Lanczos iteration 1
        initial_thetas[1] -> Lanczos iteration 2
        etc.

        When None, the starting parameters are generated randomly,
        matching the normal Qulacs workflow.

    Returns
    -------
    a : np.ndarray
        Diagonal Lanczos coefficients.

    b : np.ndarray
        Off-diagonal Lanczos coefficients.

    iteration_results : dict
        Optimizer information for each Lanczos iteration.

    ut : list[np.ndarray]
        Raw Qiskit Krylov statevectors u0, u1, ...
    """

    # ========================================================
    # INITIAL STATE
    # ========================================================

    u = np.asarray(
        u.data if isinstance(u, Statevector) else u,
        dtype=np.complex128,
    )

    n_qubits = int(
        np.log2(len(u))
    )

    n_edges = (
        connected_graphs[
            "graph_full"
        ].number_of_edges()
    )

    # ========================================================
    # CHARGE / SPIN SECTOR
    # ========================================================

    n_up = (
        charge_sec + spin_sec
    ) // 2

    n_down = (
        charge_sec - spin_sec
    ) // 2

    initial_occupations_indices = (
        get_initial_occupations_indices_qiskit(
            up_qubit_indices,
            down_qubit_indices,
            n_up,
            n_down,
        )
    )

    compilation = "generic"

    # ========================================================
    # LANCZOS COEFFICIENT ARRAYS
    # ========================================================

    a = np.zeros(
        niter + 1,
        dtype=np.complex128,
    )

    b = np.zeros(
        niter + 1,
        dtype=np.complex128,
    )

    # Keep Krylov states in raw Qiskit/Qulacs circuit ordering.
    ut = [
        u.copy()
    ]

    b[0] = 0.0 + 0.0j

    # ========================================================
    # INITIAL a0
    #
    # OpenFermion matrix multiplication requires ordering
    # conversion. The raw state stored in ut is NOT converted.
    # ========================================================

    u0_ordered = (
        qulacs_to_python_ordering_qiskit(
            ut[0],
            n_qubits,
        )
    )

    a[0] = np.vdot(
        u0_ordered,
        Hmat @ u0_ordered,
    )

    # ========================================================
    # OPTIMIZER RESULTS
    # ========================================================

    iteration_results = {}

    # ========================================================
    # LANCZOS LOOP
    # ========================================================

    for i in range(
        1,
        niter + 1,
    ):

        # ====================================================
        # CALCULATE b_i
        # ====================================================

        previous_ordered = (
            qulacs_to_python_ordering_qiskit(
                ut[i - 1],
                n_qubits,
            )
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

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Match original Qulacs behavior.
        #
        # Do NOT:
        #   - raise an error for negative b_squared
        #   - take abs(b_squared)
        #   - force it to zero
        #
        # Qulacs uses complex128 Lanczos coefficients, so a
        # negative real b_squared produces an imaginary b_i.
        # ----------------------------------------------------

        b_squared = np.complex128(
            b_squared
        )

        b[i] = np.sqrt(
            b_squared
        )

        if np.isclose(
            b[i],
            0.0 + 0.0j,
        ):
            break

        # ====================================================
        # NUMBER OF ANSATZ PARAMETERS
        # ====================================================

        n_params = (
            n_layers
            * (
                n_edges
                + n_qubits
            )
        )

        # ====================================================
        # INITIAL PARAMETER VECTOR
        # ====================================================

        if initial_thetas is None:

            theta_0 = np.random.uniform(
                low=0.0,
                high=2.0 * np.pi,
                size=n_params,
            )

        else:

            if len(initial_thetas) < i:

                raise ValueError(
                    "initial_thetas must contain "
                    "one parameter vector for every "
                    "requested Lanczos iteration."
                )

            theta_0 = np.asarray(
                initial_thetas[i - 1],
                dtype=float,
            ).copy()

            if theta_0.size != n_params:

                raise ValueError(
                    f"Iteration {i}: expected "
                    f"{n_params} parameters, "
                    f"received {theta_0.size}."
                )

        # ====================================================
        # COST-FUNCTION ARGUMENTS
        # ====================================================

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

        # ====================================================
        # FIRST OPTIMIZATION ATTEMPT
        # ====================================================

        lanczos_results = {}

        result = minimize(
            lanczos_cost_function_qiskit,
            theta_0,
            args=opt_args,
            method=optimizer,
            tol=conv_tol,
            options={
                "maxiter": int(maxiter),
            },
        )

        lanczos_results[
            result.fun
        ] = result

        # ====================================================
        # QULACS-STYLE RETRY LOGIC
        # ====================================================

        retries = 5

        if not result.success:

            for retry in range(retries):

                # --------------------------------------------
                # Normal production mode:
                #
                # Match original Qulacs and draw a new random
                # starting vector for each retry.
                # --------------------------------------------

                if initial_thetas is None:

                    theta_retry = (
                        np.random.uniform(
                            low=0.0,
                            high=2.0 * np.pi,
                            size=n_params,
                        )
                    )

                # --------------------------------------------
                # Controlled validation mode:
                #
                # Do not introduce a new random difference.
                # Reuse the supplied starting vector.
                # --------------------------------------------

                else:

                    theta_retry = (
                        theta_0.copy()
                    )

                retry_result = minimize(
                    lanczos_cost_function_qiskit,
                    theta_retry,
                    args=opt_args,
                    method=optimizer,
                    tol=conv_tol,
                    options={
                        "maxiter": int(
                            maxiter
                        ),
                    },
                )

                lanczos_results[
                    retry_result.fun
                ] = retry_result

                # --------------------------------------------
                # Qulacs behavior:
                # successful retry becomes the result.
                # --------------------------------------------

                if retry_result.success:

                    result = retry_result
                    break

                # --------------------------------------------
                # If retries continue failing, propagate the
                # lowest-cost failed optimization.
                # --------------------------------------------

                else:

                    min_lanczos_fun = min(
                        lanczos_results.keys()
                    )

                    result = (
                        lanczos_results[
                            min_lanczos_fun
                        ]
                    )

        # ====================================================
        # BUILD OPTIMIZED KRYLOV STATE
        # ====================================================

        circuit = (
            all_symmetry_ansatzae_qiskit(
                theta=result.x,
                n_qubits=n_qubits,
                n_layers=n_layers,
                initial_occupations_indices=(
                    initial_occupations_indices
                ),
                connected_graphs=(
                    connected_graphs
                ),
                compilation=compilation,
            )
        )

        new_state = (
            Statevector
            .from_instruction(
                circuit
            )
            .data
        )

        # ====================================================
        # NORMALIZE
        # ====================================================

        norm = np.linalg.norm(
            new_state
        )

        if np.isclose(
            norm,
            0.0,
        ):

            raise ValueError(
                f"Zero-norm Krylov state "
                f"encountered at iteration {i}."
            )

        new_state = (
            new_state / norm
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Store state in RAW Qiskit/Qulacs circuit ordering.
        # ----------------------------------------------------

        ut.append(
            new_state.copy()
        )

        # ====================================================
        # CALCULATE a_i
        #
        # Convert ordering ONLY for matrix multiplication.
        # ====================================================

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

        # ====================================================
        # STORE OPTIMIZER INFORMATION
        # ====================================================

        iteration_results[i] = {

            "message":
                result.message,

            "success":
                result.success,

            "status":
                result.status,

            "fun":
                result.fun,

            "parameters":
                result.x.copy(),

            "initial_theta":
                theta_0.copy(),

            "number_of_attempts":
                len(lanczos_results),

            "all_attempt_costs":
                list(
                    lanczos_results.keys()
                ),
        }

    # ========================================================
    # RETURN
    # ========================================================

    return (
        a,
        b,
        iteration_results,
        ut,
    )


def vqe_gf_pm_qiskit(
    g_vqe,
    phi_norm,
    w,
    **phi_lanczos_kwargs,
):
    """
    Qiskit equivalent of vqe.vqe_gf_pm().

    Runs the Qiskit variational Lanczos routine for either the
    particle-addition or particle-removal branch, then evaluates the
    corresponding continued-fraction Green's-function contribution.
    """

    g_vqe = np.asarray(g_vqe, dtype=np.complex128)
    w = np.asarray(w, dtype=np.complex128)

    if not np.isclose(phi_norm, 0.0):
        (
            a_vqe,
            b_vqe,
            lanczos_iterations_results,
            krylov_states,
        ) = vqe_ideal_lanczos_iterations_qiskit(
            **phi_lanczos_kwargs
        )

        g_vqe += (
            phi_norm**2
            * continued_fraction_qiskit(
                w,
                a_vqe,
                b_vqe,
            )
        )

    else:
        niter = phi_lanczos_kwargs["niter"]

        a_vqe = np.zeros(
            niter + 1,
            dtype=np.complex128,
        )

        b_vqe = np.zeros(
            niter + 1,
            dtype=np.complex128,
        )

        lanczos_iterations_results = {}
        krylov_states = []

    return (
        a_vqe,
        b_vqe,
        lanczos_iterations_results,
        g_vqe,
        krylov_states,
    )


def calculate_gf_vqe_qiskit(
    Hmat,
    H2mat,
    phi_minus,
    phi_plus,
    vqe_charge_minus: int,
    vqe_charge_plus: int,
    vqe_spin_minus: int,
    vqe_spin_plus: int,
    up_qubit_indices: list,
    down_qubit_indices: list,
    connected_graphs: dict,
    w,
    vqe_krylov_ideal_dim_minus: int,
    vqe_krylov_ideal_dim_plus: int,
    phi_minus_norm: float,
    phi_plus_norm: float,
    vqe_depth: int,
    optimizer: str,
    conv_tol: float,
    maxiters: int,
    gf_gtol: float,
    **kwargs,
):
    """
    Qiskit equivalent of vqe.calculate_gf_vqe().

    Find the Green's function based on variational Lanczos iterations
    for phi plus or minus, using the Qiskit ansatz/statevector routines
    instead of Qulacs. Mirrors vqe.calculate_gf_vqe() structurally --
    same kwarg names and same record_keeping dict keys wherever the two
    backends share the same underlying quantity (a/b coefficients, the
    combined Green's function) -- so this can be swapped in wherever
    vqe.calculate_gf_vqe() is called (e.g. dmft.py's calculate_gf())
    without dmft.py's backend-agnostic utilities (print_record,
    save_to_file, calculate_rel_errors) needing any changes.

    :param Hmat: Shifted Hamiltonian matrix (NumPy array), from
        create_qiskit_hamiltonian_matrix().
    :param H2mat: Shifted Hamiltonian-squared matrix (NumPy array).
    :param phi_minus: Initial Krylov vector for phi minus, in raw
        Qiskit/Qulacs circuit ordering (NumPy array).
    :param phi_plus: Initial Krylov vector for phi plus, same ordering.
    :param vqe_charge_minus: Charge sector for phi minus.
    :param vqe_charge_plus: Charge sector for phi plus.
    :param vqe_spin_minus: Spin sector for phi minus.
    :param vqe_spin_plus: Spin sector for phi plus.
    :param up_qubit_indices: List of spin-up register qubit indices.
    :param down_qubit_indices: List of spin-down register qubit indices.
    :param connected_graphs: Dictionary of graph objects representing the AIM.
    :param w: Complex, linearly-spaced frequency array (with broadening).
    :param vqe_krylov_ideal_dim_minus: Ideal Krylov dimension for phi minus.
    :param vqe_krylov_ideal_dim_plus: Ideal Krylov dimension for phi plus.
    :param phi_minus_norm: Norm of phi minus.
    :param phi_plus_norm: Norm of phi plus.
    :param vqe_depth: Number of ansatz layers.
    :param optimizer: Optimizer name (e.g. "COBYLA").
    :param conv_tol: Broadly defined convergence tolerance for scipy optimizers.
    :param maxiters: Max iterations/evaluations for the optimizer.
    :param gf_gtol: Gradient norm tolerance for the GF optimizer.
    :return: (g_vqe, record_keeping) -- same shape as vqe.calculate_gf_vqe().
    """
    # Same kwarg-sharing pattern as the real calculate_gf_vqe: most
    # Lanczos keyword arguments are identical between the +/- branches.
    lanczos_iteration_minus_kwargs = {
        "Hmat": Hmat,
        "H2mat": H2mat,
        "n_layers": vqe_depth,
        "up_qubit_indices": up_qubit_indices,
        "down_qubit_indices": down_qubit_indices,
        "connected_graphs": connected_graphs,
        "conv_tol": conv_tol,
        "optimizer": optimizer,
        "maxiter": maxiters,
        "gf_gtol": gf_gtol,
    }
    lanczos_iteration_plus_kwargs = {
        "Hmat": Hmat,
        "H2mat": H2mat,
        "n_layers": vqe_depth,
        "up_qubit_indices": up_qubit_indices,
        "down_qubit_indices": down_qubit_indices,
        "connected_graphs": connected_graphs,
        "conv_tol": conv_tol,
        "optimizer": optimizer,
        "maxiter": maxiters,
        "gf_gtol": gf_gtol,
    }

    phi_minus_lanczos_kwargs = dict(
        {"niter": vqe_krylov_ideal_dim_minus, "u": phi_minus, "charge_sec": vqe_charge_minus,
         "spin_sec": vqe_spin_minus},
        **lanczos_iteration_minus_kwargs,
    )
    phi_plus_lanczos_kwargs = dict(
        {"niter": vqe_krylov_ideal_dim_plus, "u": phi_plus, "charge_sec": vqe_charge_plus,
         "spin_sec": vqe_spin_plus},
        **lanczos_iteration_plus_kwargs,
    )

    g_vqe_plus = np.zeros(len(w), dtype=np.complex128)
    g_vqe_minus = np.zeros(len(w), dtype=np.complex128)

    # Same sign convention as vqe.calculate_gf_vqe: removal branch gets
    # w=-w, addition branch gets plain w, and the two are COMBINED via
    # subtraction (g_vqe = g_vqe_plus - g_vqe_minus) -- NOT addition.
    # This is the exact convention confirmed against vqe.py's source
    # during the Qiskit/Qulacs spectral-function debugging: negating
    # w for the minus branch already encodes the sign, so combining via
    # subtraction here reproduces the same physics vqe.calculate_gf_vqe
    # produces. Do not also separately negate g_vqe_minus -- that would
    # double the sign flip.
    (
        a_minus_vqe, b_minus_vqe, lanczos_iterations_results_minus,
        g_vqe_minus, krylov_states_minus,
    ) = vqe_gf_pm_qiskit(g_vqe_minus, phi_minus_norm, -w, **phi_minus_lanczos_kwargs)

    (
        a_plus_vqe, b_plus_vqe, lanczos_iterations_results_plus,
        g_vqe_plus, krylov_states_plus,
    ) = vqe_gf_pm_qiskit(g_vqe_plus, phi_plus_norm, w, **phi_plus_lanczos_kwargs)

    # Combine the two contributions of the VQE Green's function
    g_vqe = g_vqe_plus - g_vqe_minus

    ###
    record_keeping = {}
    record_keeping["lanczos_iteration_results_minus_vqe"] = lanczos_iterations_results_minus
    record_keeping["a_minus_vqe_real"] = [a.real for a in a_minus_vqe]
    record_keeping["a_minus_vqe_imag"] = [a.imag for a in a_minus_vqe]
    record_keeping["b_minus_vqe_real"] = [b.real for b in b_minus_vqe]
    record_keeping["b_minus_vqe_imag"] = [b.imag for b in b_minus_vqe]
    record_keeping["lanczos_iteration_results_plus_vqe"] = lanczos_iterations_results_plus
    record_keeping["a_plus_vqe_real"] = [a.real for a in a_plus_vqe]
    record_keeping["a_plus_vqe_imag"] = [a.imag for a in a_plus_vqe]
    record_keeping["b_plus_vqe_real"] = [b.real for b in b_plus_vqe]
    record_keeping["b_plus_vqe_imag"] = [b.imag for b in b_plus_vqe]
    # Total Green's function record keeping
    record_keeping["g_vqe_real"] = [g.real for g in g_vqe]
    record_keeping["g_vqe_imag"] = [g.imag for g in g_vqe]
    # Extra info the Qiskit backend provides that the Qulacs version
    # doesn't return (vqe_gf_pm_qiskit also hands back the optimized
    # Krylov states themselves) -- kept under Qiskit-specific keys so
    # it doesn't collide with anything dmft.py's shared utilities read.
    record_keeping["krylov_states_minus_qiskit"] = krylov_states_minus
    record_keeping["krylov_states_plus_qiskit"] = krylov_states_plus
    ###

    return g_vqe, record_keeping


def ground_state_cost_function_qiskit(
    params,
    Hmat,
    n_qubits,
    n_layers,
    initial_occupations_indices,
    connected_graphs,
    compilation="generic",
):
    """
    Qiskit equivalent of SymmQulacsVqeEmulator.expectation_value() as
    used inside solve_ground_state_local(): build the ansatz circuit
    for the given parameters/sector, and return <psi(theta)|H|psi(theta)>.

    Ordering note: unlike lanczos_cost_function_qiskit (which needs the
    OpenFermion-ordering conversion because it multiplies against a
    matrix built by get_sparse_operator), this only needs Hmat itself,
    which was ALSO built via get_sparse_operator -- so both the
    statevector and Hmat need to be in the same (OpenFermion) ordering
    for this dot product to be meaningful. The raw Qiskit statevector is
    therefore reordered before computing the expectation value, exactly
    as vqe_gf_pm_qiskit/lanczos_cost_function_qiskit do before any H or
    H^2 matrix multiplication.

    :return: real-valued energy expectation (float), suitable as a
        scipy.optimize.minimize cost function.
    """
    circuit = all_symmetry_ansatzae_qiskit(
        theta=params,
        n_qubits=n_qubits,
        n_layers=n_layers,
        initial_occupations_indices=initial_occupations_indices,
        connected_graphs=connected_graphs,
        compilation=compilation,
    )

    state_raw = Statevector.from_instruction(circuit).data
    state_ordered = qulacs_to_python_ordering_qiskit(state_raw, n_qubits)

    energy = expectation_value(state_ordered, Hmat)
    return float(np.real(energy))


def optimize_sector_qiskit(
    n_qubits,
    Hmat,
    n_layers,
    connected_graphs,
    initial_occupations_indices,
    optimizer="COBYLA",
    gtol=1e-6,
    maxiter=10000,
    compilation="generic",
):
    """
    Qiskit equivalent of SymmQulacsVqeEmulator.solve_ground_state_local():
    optimize the ansatz energy for ONE fixed charge/spin sector (i.e.
    one fixed initial_occupations_indices), starting from a single
    random theta_0. Mirrors the original's single-attempt behavior
    (no multi-start here) -- this function is a 1:1 port, not an
    enhancement.

    n_params uses the SAME formula as the real solve_ground_state_local:
        n_edges = sum(len(connected_graphs[k].edges())
                       for k in ["graph_up", "graph_down", "graph_stitch"])
        n_params = n_layers * (n_edges + n_qubits)

    NOTE: the original Qulacs solve_ground_state_local passes
    options={'maxiter': 1e9} -- a Python float. That's a real,
    independently-confirmed bug against newer scipy (COBYLA's pyprima
    backend requires a strict int and raises TypeError on a float
    maxiter/maxfun). Since this is new code, not a port of that exact
    line, maxmaxiter here is a real int with a sane default (10000)
    instead of reproducing that bug.

    :return: scipy.optimize.OptimizeResult from the sector's optimization.
    """
    n_edges = sum(
        len(connected_graphs[k].edges()) for k in ["graph_up", "graph_down", "graph_stitch"]
    )
    n_params = n_layers * (n_edges + n_qubits)

    theta_0 = np.random.uniform(low=0, high=2 * np.pi, size=n_params)

    opt_args = (Hmat, n_qubits, n_layers, initial_occupations_indices, connected_graphs, compilation)

    result = minimize(
        ground_state_cost_function_qiskit,
        theta_0,
        args=opt_args,
        method=optimizer,
        tol=gtol,
        options={"maxiter": int(maxiter)},
    )

    return result


def solve_vqe_qiskit(
    test_model,
    vqe_depth,
    optimizer,
    gs_gtol,
    checkpoint_file=None,
    results_file=None,
    display=False,
):
    """
    Qiskit equivalent of vqe.solve_vqe(): search all allowed charge/spin
    sectors, optimize the ansatz energy in each (via
    optimize_sector_qiskit), and return the lowest-energy sector's
    results. Same sector-enumeration logic as
    anderson_impurity_model.py's symmetric_ansatz_test() -- same two
    charge-sector loops (particle number <= n_qubits//2, and its
    particle-hole mirror above that), same up-down-symmetric restriction
    to S_z <= 0, same equispaced initial_occupations_indices
    construction (reused directly from get_initial_occupations_indices_qiskit,
    which already implements this).

    Graph reconstruction: like solve_ground_state_local, this derives
    connected_graphs from test_model directly (return_up_and_down_indices()
    + get_first_imp_orbital_idx()), since only test_model is given here,
    not an explicit connected_graphs. This is the same derivation
    confirmed correct/self-consistent during the Qiskit/Qulacs spectral-
    function validation earlier in this project.

    CHECKPOINTING: checkpoint_file/results_file are accepted for
    signature compatibility with vqe.solve_vqe(), but are NOT wired to
    any disk I/O here -- this function always does a full, in-memory
    sector search. This is a deliberate choice, not an oversight: the
    pickle-based checkpoint/results caching in the original Qulacs path
    was the direct cause of a long, hard-to-diagnose bug earlier in this
    project (stale cached sector results silently loaded instead of
    being recomputed, for days of debugging). If file-based caching is
    genuinely needed here later, it should use a simple, inspectable
    format (e.g. JSON keyed by (spin, charge)) with an explicit,
    visible "loaded N stale sectors from cache" message -- not silent
    pickle loading.

    :return: (vqe_gs_energy, vqe_charge, vqe_spin, minimum_angles, record_keeping)
        -- same shape as vqe.solve_vqe().
    """
    if checkpoint_file is not None or results_file is not None:
        print(
            "NOTE: solve_vqe_qiskit() does not implement checkpoint/results "
            "file caching (see docstring) -- checkpoint_file/results_file "
            "are accepted but ignored. Running a full in-memory sector search."
        )

    up_qubit_indices, down_qubit_indices = test_model.return_up_and_down_indices()
    impurity_orbital_idx = test_model.get_first_imp_orbital_idx()  # [idx], length-1 list

    n_bath_n_imp_tup = (
        len(up_qubit_indices) - len(impurity_orbital_idx),
        len(impurity_orbital_idx),
    )
    n_site_model_idx = AIMSiteModelsEnum(n_bath_n_imp_tup).create_n_site_model_idx()
    connected_graphs = create_connected_graphs(
        n_site_model_idx=n_site_model_idx, show_sub_graphs=False, show_full_plot=False,
    )

    qubit_hamiltonian = test_model.construct_qubit_hamiltonian()
    # VQE_GS_ENERGY=0.0 here: this is only used to size Hmat via
    # create_qiskit_hamiltonian_matrix's shift, which doesn't matter for
    # comparing energies WITHIN this search (every sector is shifted by
    # the same constant) -- only the resulting minimum_sector/minimum_angles
    # are used downstream, not this function's absolute energy scale.
    Hmat, _, n_qubits = create_qiskit_hamiltonian_matrix(qubit_hamiltonian, 0.0)

    sector_to_energy = {}
    sector_to_result = {}

    def _run_sector(n_electrons, z_spin):
        if z_spin > 0:
            return  # up-down symmetric case: only search S_z <= 0
        n_up = (n_electrons + z_spin) // 2
        n_down = (n_electrons - z_spin) // 2
        initial_occupations_indices = get_initial_occupations_indices_qiskit(
            up_qubit_indices, down_qubit_indices, n_up, n_down,
        )
        if display:
            print(f"Sector: n_electrons={n_electrons}, z_spin={z_spin}, "
                  f"init_occ={initial_occupations_indices}")
        result = optimize_sector_qiskit(
            n_qubits=n_qubits, Hmat=Hmat, n_layers=vqe_depth,
            connected_graphs=connected_graphs,
            initial_occupations_indices=initial_occupations_indices,
            optimizer=optimizer, gtol=gs_gtol,
        )
        sector_to_energy[(z_spin, n_electrons)] = result.fun
        sector_to_result[(z_spin, n_electrons)] = result
        if display:
            print(f"  -> energy = {result.fun:.6f}, success = {result.success}")

    # Charge sector (N) <= n_qubits // 2
    for n_electrons in range(0, n_qubits // 2 + 1):
        for z_spin in range(-n_electrons, n_electrons + 2, 2):
            _run_sector(n_electrons, z_spin)

    # Charge sector (N) > n_qubits // 2 (particle-hole mirror)
    for n_electrons in range(n_qubits // 2 + 1, n_qubits + 1):
        n_mirror = n_qubits - n_electrons
        for z_spin in range(-n_mirror, n_mirror + 1, 2):
            _run_sector(n_electrons, z_spin)

    minimum_sector = min(sector_to_energy, key=sector_to_energy.get)
    minimum_result = sector_to_result[minimum_sector]
    minimum_energy = sector_to_energy[minimum_sector]
    minimum_angles = minimum_result.x

    vqe_spin = int(np.round(minimum_sector[0]))
    vqe_charge = int(np.round(minimum_sector[1]))
    vqe_nu = (vqe_charge + vqe_spin) // 2
    vqe_nd = (vqe_charge - vqe_spin) // 2

    record_keeping = {
        "local_vqe_success": minimum_result.success,
        "local_vqe_nparams": len(minimum_angles),
        "local_vqe_nfev": minimum_result.nfev,
        # nit/njev may not exist for derivative-free optimizers like
        # COBYLA -- default to 0 rather than crash, same reasoning as
        # the scipy-compatibility patches used elsewhere in this project.
        "local_vqe_nfev_total": sum(r.nfev for r in sector_to_result.values()),
        "vqe_gs_energy": minimum_energy,
        "vqe_charge": vqe_charge,
        "vqe_spin": vqe_spin,
        "vqe_nu": vqe_nu,
        "vqe_nd": vqe_nd,
        "sector_to_energy": sector_to_energy,
    }

    if display:
        print(f"Minimum sector (spin, charge) = {minimum_sector}, energy = {minimum_energy:.6f}")

    return minimum_energy, vqe_charge, vqe_spin, minimum_angles, record_keeping
