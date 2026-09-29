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

from scipy.optimize import minimize, Bounds

from qiskit.quantum_info import Statevector
from qiskit.circuit import ParameterVector

# Optional faster statevector backend.
# Falls back to Statevector.from_instruction if qiskit-aer
# is not installed.
try:
    from qiskit_aer import AerSimulator
except ImportError:
    AerSimulator = None


from qiskit_port.qiskit_ansatz import (
    all_symmetry_ansatzae_qiskit,
)

from qiskit_port.qiskit_utils import (
    expectation_value,
    overlap,
    transition_amplitude,
    continued_fraction_qiskit,
    qulacs_to_python_ordering_qiskit,
    create_qiskit_hamiltonian_matrix,
    get_initial_occupations_indices_qiskit,
)

from n_site_graph_creation import (
    create_connected_graphs,
    AIMSiteModelsEnum,
)


# ============================================================
# FAST STATEVECTOR BACKEND
#
# The mathematical ansatz is unchanged.
#
# Instead of rebuilding a fully numeric QuantumCircuit and calling
# Statevector.from_instruction() thousands/millions of times, cache
# one parameterized circuit for each ansatz configuration and bind
# new theta values into it.
# ============================================================

_AER_STATEVECTOR_BACKEND = (
    AerSimulator(method="statevector")
    if AerSimulator is not None
    else None
)

_ANSATZ_TEMPLATE_CACHE = {}


def _connected_graph_signature(
    connected_graphs,
):
    """
    Hashable signature that preserves the SAME edge iteration order
    used by all_symmetry_ansatzae_qiskit().
    """
    return tuple(
        (
            graph_name,
            tuple(
                (
                    int(edge[0]),
                    int(edge[1]),
                )
                for edge in connected_graphs[
                    graph_name
                ].edges()
            ),
        )
        for graph_name in (
            "graph_up",
            "graph_down",
            "graph_stitch",
        )
    )


def _ansatz_cache_key(
    n_qubits,
    n_layers,
    initial_occupations_indices,
    connected_graphs,
    compilation,
):
    return (
        int(n_qubits),
        int(n_layers),
        tuple(
            int(x)
            for x in initial_occupations_indices
        ),
        _connected_graph_signature(
            connected_graphs
        ),
        str(compilation),
    )


def _get_parameterized_ansatz_template(
    n_qubits,
    n_layers,
    initial_occupations_indices,
    connected_graphs,
    compilation="generic",
):
    """
    Build a parameterized ansatz once and cache it.
    """
    key = _ansatz_cache_key(
        n_qubits=n_qubits,
        n_layers=n_layers,
        initial_occupations_indices=(
            initial_occupations_indices
        ),
        connected_graphs=connected_graphs,
        compilation=compilation,
    )

    if key in _ANSATZ_TEMPLATE_CACHE:
        return _ANSATZ_TEMPLATE_CACHE[key]

    n_edges = sum(
        connected_graphs[name].number_of_edges()
        for name in (
            "graph_up",
            "graph_down",
            "graph_stitch",
        )
    )

    n_params = (
        n_layers
        * (
            n_edges
            + n_qubits
        )
    )

    parameters = ParameterVector(
        "theta",
        n_params,
    )

    circuit = all_symmetry_ansatzae_qiskit(
        theta=parameters,
        n_qubits=n_qubits,
        n_layers=n_layers,
        initial_occupations_indices=(
            initial_occupations_indices
        ),
        connected_graphs=connected_graphs,
        compilation=compilation,
    )

    if _AER_STATEVECTOR_BACKEND is not None:
        circuit.save_statevector()

    _ANSATZ_TEMPLATE_CACHE[key] = (
        circuit,
        parameters,
    )

    return circuit, parameters


def _simulate_ansatz_state_qiskit(
    theta,
    n_qubits,
    n_layers,
    initial_occupations_indices,
    connected_graphs,
    compilation="generic",
):
    """
    Return the RAW Qiskit-order statevector for the symmetry-preserving
    ansatz.

    Aer is used when available. Statevector.from_instruction remains
    as a compatibility fallback.
    """
    theta = np.asarray(
        theta,
        dtype=float,
    )

    if _AER_STATEVECTOR_BACKEND is None:

        circuit = all_symmetry_ansatzae_qiskit(
            theta=theta,
            n_qubits=n_qubits,
            n_layers=n_layers,
            initial_occupations_indices=(
                initial_occupations_indices
            ),
            connected_graphs=connected_graphs,
            compilation=compilation,
        )

        return (
            Statevector
            .from_instruction(circuit)
            .data
            .copy()
        )

    circuit, parameters = (
        _get_parameterized_ansatz_template(
            n_qubits=n_qubits,
            n_layers=n_layers,
            initial_occupations_indices=(
                initial_occupations_indices
            ),
            connected_graphs=connected_graphs,
            compilation=compilation,
        )
    )

    if theta.size != len(parameters):
        raise ValueError(
            f"Expected {len(parameters)} ansatz parameters, "
            f"received {theta.size}."
        )

    parameter_map = {
        parameter: float(value)
        for parameter, value
        in zip(parameters, theta)
    }

    bound_circuit = circuit.assign_parameters(
        parameter_map,
        inplace=False,
    )

    result = _AER_STATEVECTOR_BACKEND.run(
        bound_circuit
    ).result()

    state = np.asarray(
        result.data(0)["statevector"],
        dtype=np.complex128,
    )

    return state.copy()

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
    initial_occupations_indices = (
        get_initial_occupations_indices_qiskit(
            up_qubit_indices,
            down_qubit_indices,
            vqe_nu,
            vqe_nd,
        )
    )

    state_array = _simulate_ansatz_state_qiskit(
        theta=minimum_angles,
        n_qubits=n_qubits,
        n_layers=vqe_depth,
        initial_occupations_indices=(
            initial_occupations_indices
        ),
        connected_graphs=connected_graphs,
        compilation="generic",
    )

    statevector = Statevector(
        state_array
    )

    return (
        statevector,
        state_array.copy(),
    )


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
    h_previous_raw=None,
):
    """
    Qiskit equivalent of vqe.lanczos_cost_function().

    Krylov vectors remain in RAW Qiskit/Qulacs ordering.

    The optional h_previous_raw contains

        P H P |u_(i-1)>

    where P is the bit-reversal permutation. Since this quantity is
    invariant during one optimizer run, calculating it once outside
    scipy.optimize avoids repeating ordering conversion and H @ u on
    every cost-function evaluation.
    """

    ws_raw = _simulate_ansatz_state_qiskit(
        theta=params,
        n_qubits=n_qubits,
        n_layers=n_layers,
        initial_occupations_indices=(
            initial_occupations_indices
        ),
        connected_graphs=connected_graphs,
        compilation=compilation,
    )

    if h_previous_raw is None:

        previous_ordered = (
            qulacs_to_python_ordering_qiskit(
                ut[i - 1],
                n_qubits,
            )
        )

        h_previous_ordered = (
            Hmat
            @ previous_ordered
        )

        h_previous_raw = (
            qulacs_to_python_ordering_qiskit(
                h_previous_ordered,
                n_qubits,
            )
        )

    # Equivalent to
    #
    # <P ws | H | P u_(i-1)> - b_i
    #
    # because P is a self-inverse permutation.
    e1 = abs(
        np.vdot(
            ws_raw,
            h_previous_raw,
        )
        - b[i]
    )

    e2 = abs(
        np.vdot(
            ws_raw,
            ut[i - 1],
        )
    )

    if i > 1:
        e3 = abs(
            np.vdot(
                ws_raw,
                ut[i - 2],
            )
        )
    else:
        e3 = 0.0

    e4 = 0.0

    if i == 4:
        e4 += abs(
            np.vdot(
                ws_raw,
                ut[1],
            )
        ) ** 2

    if i >= 5:
        for j in range(3, 6):
            e4 += abs(
                np.vdot(
                    ws_raw,
                    ut[i - j],
                )
            ) ** 2

    cost = (
        e1**2
        + e2**2
        + e3**2
        + e4
    )

    return float(
        np.real(cost)
    )


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
    optimizer="BFGS",
    maxiter=1_000_000,
    gf_gtol=5e-5,
    initial_thetas=None,
    display=False,
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
        Gradient-norm tolerance used by the BFGS Green's-function optimizer.

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
        
        # ============================================================
        # H^2 expectation used to calculate b_i
        # ============================================================
        
        h2_expectation = np.vdot(
            previous_ordered,
            H2mat @ previous_ordered,
        )
        
        
        # ============================================================
        # H |u_(i-1)> used repeatedly by the optimizer cost function
        #
        # This does NOT depend on theta, so calculate it ONCE here
        # instead of thousands of times inside the cost function.
        # ============================================================
        
        h_previous_ordered = (
            Hmat @ previous_ordered
        )
        
        # Convert back to RAW Qiskit/Qulacs circuit ordering,
        # because ws and ut are stored in raw ordering.
        h_previous_raw = (
            qulacs_to_python_ordering_qiskit(
                h_previous_ordered,
                n_qubits,
            )
        )
        
        
        # ============================================================
        # Calculate b_i
        # ============================================================
        
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
        
        
        # Preserve complex behavior while removing only tiny
        # floating-point imaginary noise.
        b_squared = np.complex128(
            b_squared
        )
        
        if abs(b_squared.imag) < 1e-12:
        
            b_squared = np.complex128(
                b_squared.real + 0.0j
            )
        
        
        b[i] = np.sqrt(
            b_squared
        )

        if display:
            print(
                f"Lanczos {i}/{niter}: "
                f"a_prev={a[i - 1]:.8g}, "
                f"b={b[i]:.8g}"
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
            h_previous_raw,
        )

       
        # ====================================================
        # FIRST OPTIMIZATION ATTEMPT
        # ====================================================
        
        lanczos_results = {}
        
        # ----------------------------------------------------
        # Match optimizer-specific settings used by the
        # original Qulacs AIM implementation.
        #
        # For BFGS:
        #   gtol = Green's-function gradient tolerance
        #   eps  = finite-difference step size
        #
        # Do not pass gtol/eps to optimizers such as COBYLA.
        # ----------------------------------------------------
        
        optimizer_options = {
            "maxiter": int(maxiter),
        }
        
        if optimizer.upper() == "BFGS":
        
            optimizer_options.update(
                {
                    "gtol": gf_gtol,
                    "eps": 1e-7,
                }
            )
        
        
        result = minimize(
            lanczos_cost_function_qiskit,
            theta_0,
            args=opt_args,
            method=optimizer,
            tol=conv_tol,
            options=optimizer_options,
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
                    options=optimizer_options,
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

        new_state = _simulate_ansatz_state_qiskit(
            theta=result.x,
            n_qubits=n_qubits,
            n_layers=n_layers,
            initial_occupations_indices=(
                initial_occupations_indices
            ),
            connected_graphs=connected_graphs,
            compilation=compilation,
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

        if display:
            print(
                f"  optimizer: success={result.success}, "
                f"attempts={len(lanczos_results)}, "
                f"nfev={getattr(result, 'nfev', 0)}, "
                f"fun={result.fun:.6e}"
            )

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
    seed_minus: int = 0,
    seed_plus: int = 0,
    display: bool = False,
    **kwargs,
):
    """
    Qiskit equivalent of vqe.calculate_gf_vqe().

    Find the Green's function based on variational Lanczos iterations
    for phi plus or minus, using the Qiskit ansatz/statevector routines
    instead of Qulacs.

    Mirrors vqe.calculate_gf_vqe() structurally -- same kwarg names
    and same record_keeping dict keys wherever the two backends share
    the same underlying quantity.

    The removal and addition branches use independent random seeds so
    optimizer retries in one branch cannot change the random starting
    parameters used by the other branch.

    Parameters
    ----------
    Hmat
        Shifted Hamiltonian matrix.

    H2mat
        Shifted Hamiltonian-squared matrix.

    phi_minus
        Initial Krylov vector for the removal branch, in raw
        Qiskit/Qulacs circuit ordering.

    phi_plus
        Initial Krylov vector for the addition branch, in raw
        Qiskit/Qulacs circuit ordering.

    vqe_charge_minus : int
        Charge sector for phi minus.

    vqe_charge_plus : int
        Charge sector for phi plus.

    vqe_spin_minus : int
        Spin sector for phi minus.

    vqe_spin_plus : int
        Spin sector for phi plus.

    up_qubit_indices : list
        Spin-up qubit indices.

    down_qubit_indices : list
        Spin-down qubit indices.

    connected_graphs : dict
        AIM graph information.

    w
        Complex frequency grid.

    vqe_krylov_ideal_dim_minus : int
        Number of variational Lanczos iterations for removal.

    vqe_krylov_ideal_dim_plus : int
        Number of variational Lanczos iterations for addition.

    phi_minus_norm : float
        Norm of the unnormalized removal Krylov-zero state.

    phi_plus_norm : float
        Norm of the unnormalized addition Krylov-zero state.

    vqe_depth : int
        Number of ansatz layers.

    optimizer : str
        SciPy optimizer name.

    conv_tol : float
        General optimizer convergence tolerance.

    maxiters : int
        Maximum optimizer iterations/evaluations.

    gf_gtol : float
        Gradient-norm tolerance for the Green's-function optimizer.

    seed_minus : int, default=0
        Random seed used for the removal branch.

    seed_plus : int, default=0
        Random seed used for the addition branch.

    Returns
    -------
    g_vqe
        Combined variational Green's function.

    record_keeping : dict
        Lanczos coefficients, optimizer information, Green's function,
        and Qiskit Krylov states.
    """

    # ========================================================
    # SHARED LANCZOS SETTINGS
    # ========================================================

    lanczos_iteration_minus_kwargs = {
        "display": display,
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
        "display": display,
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


    # ========================================================
    # REMOVAL-BRANCH SETTINGS
    # ========================================================

    phi_minus_lanczos_kwargs = dict(
        {
            "niter":
                vqe_krylov_ideal_dim_minus,

            "u":
                phi_minus,

            "charge_sec":
                vqe_charge_minus,

            "spin_sec":
                vqe_spin_minus,
        },
        **lanczos_iteration_minus_kwargs,
    )


    # ========================================================
    # ADDITION-BRANCH SETTINGS
    # ========================================================

    phi_plus_lanczos_kwargs = dict(
        {
            "niter":
                vqe_krylov_ideal_dim_plus,

            "u":
                phi_plus,

            "charge_sec":
                vqe_charge_plus,

            "spin_sec":
                vqe_spin_plus,
        },
        **lanczos_iteration_plus_kwargs,
    )


    # ========================================================
    # GREEN'S-FUNCTION ARRAYS
    # ========================================================

    g_vqe_plus = np.zeros(
        len(w),
        dtype=np.complex128,
    )

    g_vqe_minus = np.zeros(
        len(w),
        dtype=np.complex128,
    )


    # ========================================================
    # REMOVAL BRANCH
    #
    # IMPORTANT:
    #
    # Give the removal branch its own RNG starting point.
    #
    # Any retries that happen during removal may consume more
    # random numbers, but they can no longer affect the later
    # addition branch because addition will be reseeded
    # independently below.
    # ========================================================

    np.random.seed(
        seed_minus
    )

    (
        a_minus_vqe,
        b_minus_vqe,
        lanczos_iterations_results_minus,
        g_vqe_minus,
        krylov_states_minus,
    ) = vqe_gf_pm_qiskit(
        g_vqe_minus,
        phi_minus_norm,
        -w,
        **phi_minus_lanczos_kwargs,
    )


    # ========================================================
    # ADDITION BRANCH
    #
    # Reset the RNG independently of what happened during
    # removal.
    #
    # Therefore:
    #
    # removal retries
    #       DO NOT
    # change addition theta_0.
    # ========================================================

    np.random.seed(
        seed_plus
    )

    (
        a_plus_vqe,
        b_plus_vqe,
        lanczos_iterations_results_plus,
        g_vqe_plus,
        krylov_states_plus,
    ) = vqe_gf_pm_qiskit(
        g_vqe_plus,
        phi_plus_norm,
        w,
        **phi_plus_lanczos_kwargs,
    )


    # ========================================================
    # COMBINE REMOVAL + ADDITION
    #
    # Preserve the validated Qulacs sign convention:
    #
    #   removal evaluated at -w
    #   addition evaluated at +w
    #
    # followed by
    #
    #   G = G_plus - G_minus
    #
    # Do NOT separately negate g_vqe_minus here.
    # ========================================================

    g_vqe = (
        g_vqe_plus
        - g_vqe_minus
    )


    # ========================================================
    # RECORD KEEPING
    # ========================================================

    record_keeping = {}


    # --------------------------------------------------------
    # Removal
    # --------------------------------------------------------

    record_keeping[
        "lanczos_iteration_results_minus_vqe"
    ] = lanczos_iterations_results_minus

    record_keeping[
        "a_minus_vqe_real"
    ] = [
        a.real
        for a in a_minus_vqe
    ]

    record_keeping[
        "a_minus_vqe_imag"
    ] = [
        a.imag
        for a in a_minus_vqe
    ]

    record_keeping[
        "b_minus_vqe_real"
    ] = [
        b.real
        for b in b_minus_vqe
    ]

    record_keeping[
        "b_minus_vqe_imag"
    ] = [
        b.imag
        for b in b_minus_vqe
    ]


    # --------------------------------------------------------
    # Addition
    # --------------------------------------------------------

    record_keeping[
        "lanczos_iteration_results_plus_vqe"
    ] = lanczos_iterations_results_plus

    record_keeping[
        "a_plus_vqe_real"
    ] = [
        a.real
        for a in a_plus_vqe
    ]

    record_keeping[
        "a_plus_vqe_imag"
    ] = [
        a.imag
        for a in a_plus_vqe
    ]

    record_keeping[
        "b_plus_vqe_real"
    ] = [
        b.real
        for b in b_plus_vqe
    ]

    record_keeping[
        "b_plus_vqe_imag"
    ] = [
        b.imag
        for b in b_plus_vqe
    ]


    # --------------------------------------------------------
    # Total Green's function
    # --------------------------------------------------------

    record_keeping[
        "g_vqe_real"
    ] = [
        g.real
        for g in g_vqe
    ]

    record_keeping[
        "g_vqe_imag"
    ] = [
        g.imag
        for g in g_vqe
    ]


    # --------------------------------------------------------
    # Qiskit-specific optimized Krylov states
    # --------------------------------------------------------

    record_keeping[
        "krylov_states_minus_qiskit"
    ] = krylov_states_minus

    record_keeping[
        "krylov_states_plus_qiskit"
    ] = krylov_states_plus


    # --------------------------------------------------------
    # Record seeds used for reproducibility
    # --------------------------------------------------------

    record_keeping[
        "seed_minus_qiskit"
    ] = seed_minus

    record_keeping[
        "seed_plus_qiskit"
    ] = seed_plus


    # ========================================================
    # RETURN
    # ========================================================

    return (
        g_vqe,
        record_keeping,
    )


def ground_state_cost_function_qiskit(
    params,
    Hmat,
    n_qubits,
    n_layers,
    initial_occupations_indices,
    connected_graphs,
    compilation="generic",
):
    state_raw = _simulate_ansatz_state_qiskit(
        theta=params,
        n_qubits=n_qubits,
        n_layers=n_layers,
        initial_occupations_indices=(
            initial_occupations_indices
        ),
        connected_graphs=connected_graphs,
        compilation=compilation,
    )

    state_ordered = (
        qulacs_to_python_ordering_qiskit(
            state_raw,
            n_qubits,
        )
    )

    energy = expectation_value(
        state_ordered,
        Hmat,
    )

    return float(
        np.real(energy)
    )


def optimize_sector_qiskit(
    n_qubits,
    Hmat,
    n_layers,
    connected_graphs,
    initial_occupations_indices,
    optimizer="BFGS",
    gtol=5e-4,
    maxiter=1_000_000_000,
    compilation="generic",
):
    """Optimize one fixed charge/spin sector with the Qiskit ansatz.

    The parameter count and random initialization match the original
    SymmQulacsVqeEmulator.solve_ground_state_local() implementation.
    BFGS uses the original gradient tolerance; Nelder-Mead and COBYLA
    use the original tolerances.
    """
    n_edges = sum(
        connected_graphs[name].number_of_edges()
        for name in (
            "graph_up",
            "graph_down",
            "graph_stitch",
        )
    )

    n_params = n_layers * (n_edges + n_qubits)

    theta_0 = np.random.uniform(
        low=0.0,
        high=2.0 * np.pi,
        size=n_params,
    )

    opt_args = (
        Hmat,
        n_qubits,
        n_layers,
        initial_occupations_indices,
        connected_graphs,
        compilation,
    )

    method = optimizer.upper()

    if method == "BFGS":
        result = minimize(
            ground_state_cost_function_qiskit,
            theta_0,
            args=opt_args,
            method="BFGS",
            options={
                "maxiter": int(maxiter),
                "gtol": gtol,
            },
        )

    elif method == "NELDER-MEAD":
        result = minimize(
            ground_state_cost_function_qiskit,
            theta_0,
            args=opt_args,
            method="Nelder-Mead",
            tol=1e-9,
            bounds=Bounds(
                lb=np.zeros(n_params),
                ub=np.full(n_params, 2.0 * np.pi),
            ),
            options={
                "maxiter": int(maxiter),
            },
        )

    elif method == "COBYLA":
        result = minimize(
            ground_state_cost_function_qiskit,
            theta_0,
            args=opt_args,
            method="COBYLA",
            tol=1e-4,
            options={
                "maxiter": int(maxiter),
            },
        )

    else:
        raise ValueError(
            "Unsupported ground-state optimizer. "
            "Use BFGS, Nelder-Mead, or COBYLA."
        )

    return result

def solve_vqe_qiskit(
    test_model,
    vqe_depth,
    optimizer="BFGS",
    gs_gtol=5e-4,
    maxiter=1_000_000_000,
    checkpoint_file=None,
    results_file=None,
    display=False,
):
    """Search all allowed charge/spin sectors and return the VQE minimum.

    This is the Qiskit counterpart of vqe.solve_vqe() plus the sector
    enumeration performed by symmetric_ansatz_test(). File checkpointing
    is deliberately not used here; every run is explicit and in-memory.
    """
    if checkpoint_file is not None or results_file is not None:
        print(
            "NOTE: solve_vqe_qiskit() ignores checkpoint/results files "
            "and performs a fresh in-memory sector search."
        )

    up_qubit_indices, down_qubit_indices = (
        test_model.return_up_and_down_indices()
    )
    impurity_orbital_idx = test_model.get_first_imp_orbital_idx()

    n_bath_n_imp_tup = (
        len(up_qubit_indices) - len(impurity_orbital_idx),
        len(impurity_orbital_idx),
    )
    n_site_model_idx = (
        AIMSiteModelsEnum(n_bath_n_imp_tup).create_n_site_model_idx()
    )
    connected_graphs = create_connected_graphs(
        n_site_model_idx=n_site_model_idx,
        show_sub_graphs=False,
        show_full_plot=False,
    )

    qubit_hamiltonian = test_model.construct_qubit_hamiltonian()
    Hmat, _, n_qubits = create_qiskit_hamiltonian_matrix(
        qubit_hamiltonian,
        0.0,
    )

    sector_to_energy = {}
    sector_to_result = {}

    def _run_sector(n_electrons, z_spin):
        # The original symmetric search explicitly evaluates S_z <= 0
        # and uses up/down symmetry for the positive-spin counterparts.
        if z_spin > 0:
            return

        n_up = (n_electrons + z_spin) // 2
        n_down = (n_electrons - z_spin) // 2

        initial_occupations_indices = (
            get_initial_occupations_indices_qiskit(
                up_qubit_indices,
                down_qubit_indices,
                n_up,
                n_down,
            )
        )

        if display:
            print(
                f"Sector: n_electrons={n_electrons}, "
                f"z_spin={z_spin}, "
                f"init_occ={initial_occupations_indices}"
            )

        result = optimize_sector_qiskit(
            n_qubits=n_qubits,
            Hmat=Hmat,
            n_layers=vqe_depth,
            connected_graphs=connected_graphs,
            initial_occupations_indices=initial_occupations_indices,
            optimizer=optimizer,
            gtol=gs_gtol,
            maxiter=maxiter,
        )

        key = (z_spin, n_electrons)
        sector_to_energy[key] = float(result.fun)
        sector_to_result[key] = result

        if display:
            print(
                f"  -> energy={result.fun:.12f}, "
                f"success={result.success}, "
                f"nfev={getattr(result, 'nfev', 0)}"
            )

    # Charge N <= n_qubits/2.
    for n_electrons in range(0, n_qubits // 2 + 1):
        for z_spin in range(-n_electrons, n_electrons + 2, 2):
            _run_sector(n_electrons, z_spin)

    # Charge N > n_qubits/2, using the particle-hole mirror range.
    for n_electrons in range(n_qubits // 2 + 1, n_qubits + 1):
        n_mirror = n_qubits - n_electrons
        for z_spin in range(-n_mirror, n_mirror + 1, 2):
            _run_sector(n_electrons, z_spin)

    if not sector_to_energy:
        raise RuntimeError("No charge/spin sectors were optimized.")

    minimum_sector = min(
        sector_to_energy,
        key=sector_to_energy.get,
    )
    minimum_result = sector_to_result[minimum_sector]
    minimum_energy = sector_to_energy[minimum_sector]
    minimum_angles = np.asarray(minimum_result.x, dtype=float).copy()

    vqe_spin = int(np.round(minimum_sector[0]))
    vqe_charge = int(np.round(minimum_sector[1]))
    vqe_nu = (vqe_charge + vqe_spin) // 2
    vqe_nd = (vqe_charge - vqe_spin) // 2

    record_keeping = {
        "local_vqe_success": bool(minimum_result.success),
        "local_vqe_nparams": len(minimum_angles),
        "local_vqe_nfev": int(getattr(minimum_result, "nfev", 0)),
        "local_vqe_njev": int(getattr(minimum_result, "njev", 0)),
        "local_vqe_nit": int(getattr(minimum_result, "nit", 0)),
        "local_vqe_nfev_total": int(
            sum(getattr(r, "nfev", 0) for r in sector_to_result.values())
        ),
        "local_vqe_njev_total": int(
            sum(getattr(r, "njev", 0) for r in sector_to_result.values())
        ),
        "local_vqe_nit_total": int(
            sum(getattr(r, "nit", 0) for r in sector_to_result.values())
        ),
        "vqe_gs_energy": minimum_energy,
        "vqe_charge": vqe_charge,
        "vqe_spin": vqe_spin,
        "vqe_nu": vqe_nu,
        "vqe_nd": vqe_nd,
        "sector_to_energy": {
            f"spin={int(spin)},charge={int(charge)}": float(energy)
            for (spin, charge), energy in sector_to_energy.items()
        },
    }

    if display:
        print(
            f"Minimum sector (spin, charge)={minimum_sector}, "
            f"energy={minimum_energy:.12f}"
        )

    return (
        minimum_energy,
        vqe_charge,
        vqe_spin,
        minimum_angles,
        record_keeping,
    )

