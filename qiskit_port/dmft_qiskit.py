"""
dmft_qiskit.py -- high-level Qiskit driver for the AIM workflow.

Top layer of the qiskit_port package. Plays the same role dmft.py plays
for Qulacs: initialize the AIM, solve both an exact and a VQE ground
state, compare them, compute both Green's functions, compare those,
and hand back one results dictionary -- callable as a single function
from a notebook.

Every function here was written by cross-referencing the REAL dmft.py
source directly (not guessed at) -- specifically compare_gs(),
calculate_gs(), calculate_gf(), and calculate_rel_errors(). Two
deliberate, documented differences from the original:

1. initialize_system() and the exact-side calls (exact.solve_exact_gs,
   exact.construct_exact_krylov_dimensions, exact.calculate_gf_exact)
   are REUSED DIRECTLY, unchanged -- none of that touches a quantum
   circuit, so there is nothing backend-specific to port. Same for
   vqe.construct_vqe_krylov_dimensions, which is pure combinatorics
   (binomial coefficients on charge/spin sectors) with no Qulacs
   objects involved.

2. Both workflows are available: run_aim_qiskit() performs one fixed
   VQE depth, while run_gs_error_experiment_qiskit() mirrors the original
   depth-scaling driver and increases the ansatz depth until the requested
   ground-state overlap error is reached.

Sign convention note: the particle-removal/addition combination logic
does NOT live here. calculate_green_function_qiskit() calls
qv.calculate_gf_vqe_qiskit(), which already has the validated sign
convention (w=-w for removal, combine via subtraction) built in --
this file never touches a raw omega sign, by design (see the earlier
"why the sign bug happened" investigation: the bug came from that
logic being hand-copied into a calling script instead of living in one
place).
"""

import numpy as np
import matplotlib.pyplot as plt

import exact
import vqe
from dmft import initialize_system  # noqa: F401 -- re-exported, backend-agnostic, reused directly

import qiskit_port.qiskit_vqe as qv
from qiskit_port.qiskit_utils import (
    create_qiskit_hamiltonian_matrix,
    qulacs_to_python_ordering_qiskit,
)


# ============================================================
# B. compare_ground_states_qiskit()
# Direct port of dmft.compare_gs(), operating on the Qiskit state
# instead of the Qulacs one. Verified against dmft.py's actual source:
#
#     gs_energy_rel_error = |E_exact - E_vqe| / |E_exact|
#     gs_overlap = |vqe_gs^dagger . exact_gs|
#     gs_error = 1 - gs_overlap
#
# IMPORTANT: exact_gs comes out of exact diagonalization in the same
# (OpenFermion) ordering the Hamiltonian matrix uses. qiskit_state must
# be in that SAME ordering for this overlap to mean anything -- pass
# the ORDERED state (qulacs_to_python_ordering_qiskit applied), not the
# raw circuit-ordering array. calculate_ground_state_qiskit() below
# already returns both, precisely so this doesn't get mixed up.
# ============================================================

def compare_ground_states_qiskit(
    exact_energy,
    qiskit_energy,
    exact_state,
    qiskit_state_ordered,
):
    """
    Compare the ground state and ground-state energy found exactly and
    by the Qiskit VQE solver.

    :param exact_energy: Ground-state energy from exact diagonalization.
    :param qiskit_energy: Ground-state energy as found by Qiskit VQE.
    :param exact_state: Exactly solved ground state (OpenFermion ordering).
    :param qiskit_state_ordered: Qiskit VQE ground state, ALREADY reordered
        into OpenFermion convention (see note above).
    :return: dict with gs_energy_rel_error, gs_fidelity, gs_error.
    """
    gs_energy_rel_error = np.abs(exact_energy - qiskit_energy) / np.abs(exact_energy)

    gs_overlap = np.abs(np.vdot(qiskit_state_ordered, exact_state))
    gs_fidelity = gs_overlap ** 2
    gs_error = 1.0 - gs_overlap

    record_keeping = {
        "gs_overlap": gs_overlap,
        "gs_fidelity": gs_fidelity,
        "gs_energy_rel_error": gs_energy_rel_error,
        "gs_error": gs_error,
    }

    return record_keeping


# ============================================================
# C. calculate_ground_state_qiskit()
# Calls solve_vqe_qiskit() -> construct_vqe_gs_qiskit(), the two
# functions built and validated earlier this session (independent
# Qiskit-side ground-state search, no reuse of Qulacs's angles).
# ============================================================

def calculate_ground_state_qiskit(
    test_model,
    qubit_hamiltonian,
    connected_graphs,
    up_qubit_indices,
    down_qubit_indices,
    vqe_depth,
    optimizer="BFGS",
    gs_gtol=5e-4,
    maxiter=1_000_000_000,
    display=False,
):
    
    """
    Solve for the Qiskit-side VQE ground state: search all allowed
    charge/spin sectors (solve_vqe_qiskit), then rebuild the winning
    state (construct_vqe_gs_qiskit).

    :return: dict with keys energy, state (raw circuit ordering),
        state_ordered (OpenFermion ordering, for exact-state
        comparison), angles, charge, spin, n_up, n_down, record.
    """
    (
        vqe_energy,
        vqe_charge,
        vqe_spin,
        minimum_angles,
        record_keeping,
    ) = qv.solve_vqe_qiskit(
        test_model=test_model,
        vqe_depth=vqe_depth,
        optimizer=optimizer,
        gs_gtol=gs_gtol,
        maxiter=maxiter,
        display=display,
    )    

    vqe_nu = record_keeping["vqe_nu"]
    vqe_nd = record_keeping["vqe_nd"]

    Hmat, _, n_qubits = create_qiskit_hamiltonian_matrix(qubit_hamiltonian, vqe_energy)

    qiskit_state, qiskit_state_array = qv.construct_vqe_gs_qiskit(
        n_qubits=n_qubits,
        up_qubit_indices=up_qubit_indices,
        down_qubit_indices=down_qubit_indices,
        vqe_depth=vqe_depth,
        connected_graphs=connected_graphs,
        minimum_angles=minimum_angles,
        vqe_nu=vqe_nu,
        vqe_nd=vqe_nd,
    )

    qiskit_state_ordered = qulacs_to_python_ordering_qiskit(qiskit_state_array, n_qubits)

    return {
        "energy": vqe_energy,
        "state": qiskit_state_array,
        "state_ordered": qiskit_state_ordered,
        "angles": minimum_angles,
        "charge": vqe_charge,
        "spin": vqe_spin,
        "n_up": vqe_nu,
        "n_down": vqe_nd,
        "n_qubits": n_qubits,
        "record": record_keeping,
    }


# ============================================================
# D. calculate_green_function_qiskit()
# Direct port of dmft.calculate_gf()'s VQE-side logic (section 3.2 and
# 4.2 in the original), routed through the Qiskit functions. Reuses
# vqe.construct_vqe_krylov_dimensions() UNCHANGED -- confirmed against
# vqe.py's actual source that it's pure combinatorics (binomial
# coefficients on the +/-1 charge/spin sectors), no Qulacs objects
# touched anywhere in it.
# ============================================================

def calculate_green_function_qiskit(
    impurity_orbital,
    qubit_hamiltonian,
    ground_state_result,
    up_qubit_indices,
    down_qubit_indices,
    connected_graphs,
    n_orbitals,
    vqe_depth,
    w,
    optimizer,
    conv_tol,
    maxiter,
    gf_gtol,
    display=False,
    seed_minus=0,
    seed_plus=0,
):
    """
    Build the Qiskit-side Green's function: Hamiltonian matrices ->
    phi-/phi+ Krylov states -> removal/addition sectors -> Lanczos on
    each branch (via calculate_gf_vqe_qiskit, which already has the
    validated sign convention baked in) -> combined G(omega) ->
    spectral function.

    :param ground_state_result: the dict returned by
        calculate_ground_state_qiskit().
    :return: dict with g_minus, g_plus, g_total, spectral, a/b
        coefficients for both branches, phi norms, and Krylov
        dimensions actually used.
    """
    vqe_energy = ground_state_result["energy"]
    vqe_charge = ground_state_result["charge"]
    vqe_spin = ground_state_result["spin"]
    minimum_angles = ground_state_result["angles"]
    n_qubits = ground_state_result["n_qubits"]

    Hmat, H2mat, n_qubits_check = create_qiskit_hamiltonian_matrix(qubit_hamiltonian, vqe_energy)
    assert n_qubits == n_qubits_check, "Qubit count changed between ground-state and GF steps"

    # Rebuild the ground state at this (freshly re-shifted) Hamiltonian,
    # exactly mirroring calculate_gf()'s "recreating gs for green's
    # function determination" step.
    qiskit_gs, qiskit_gs_array = qv.construct_vqe_gs_qiskit(
        n_qubits=n_qubits,
        up_qubit_indices=up_qubit_indices,
        down_qubit_indices=down_qubit_indices,
        vqe_depth=vqe_depth,
        connected_graphs=connected_graphs,
        minimum_angles=minimum_angles,
        vqe_nu=ground_state_result["n_up"],
        vqe_nd=ground_state_result["n_down"],
    )

    # Sectors + ideal Krylov dimensions -- backend-agnostic, reused directly.
    (
        vqe_nu_minus, vqe_nd_minus, charge_minus, spin_minus, krylov_dim_minus,
        vqe_nu_plus, vqe_nd_plus, charge_plus, spin_plus, krylov_dim_plus,
        krylov_dim_record,
    ) = vqe.construct_vqe_krylov_dimensions(
        up_idx=up_qubit_indices, impurity_orbital=impurity_orbital,
        n_orbitals=n_orbitals, vqe_charge=vqe_charge, vqe_spin=vqe_spin,
    )

    # Krylov zero states (Qiskit-side)
    (
        qiskit_phi_minus, qiskit_phi_minus_array, qiskit_phi_minus_norm,
        qiskit_phi_plus, qiskit_phi_plus_array, qiskit_phi_plus_norm,
        krylov_zero_record,
    ) = qv.vqe_krylov_zero_state_qiskit(
        vqe_gs=qiskit_gs, impurity_orbital=impurity_orbital, n_qubits=n_qubits,
    )

    # Green's function -- calculate_gf_vqe_qiskit already has the
    # validated removal/addition sign convention built in.
    g_total, gf_record = qv.calculate_gf_vqe_qiskit(
        Hmat=Hmat,
        H2mat=H2mat,
        phi_minus=qiskit_phi_minus_array,
        phi_plus=qiskit_phi_plus_array,
        vqe_charge_minus=charge_minus,
        vqe_charge_plus=charge_plus,
        vqe_spin_minus=spin_minus,
        vqe_spin_plus=spin_plus,
        up_qubit_indices=up_qubit_indices,
        down_qubit_indices=down_qubit_indices,
        connected_graphs=connected_graphs,
        w=w,
        vqe_krylov_ideal_dim_minus=krylov_dim_minus,
        vqe_krylov_ideal_dim_plus=krylov_dim_plus,
        phi_minus_norm=qiskit_phi_minus_norm,
        phi_plus_norm=qiskit_phi_plus_norm,
        vqe_depth=vqe_depth,
        optimizer=optimizer,
        conv_tol=conv_tol,
        maxiters=maxiter,
        gf_gtol=gf_gtol,
        display=display,
        seed_minus=seed_minus,
        seed_plus=seed_plus,
    )

    spectral = -(1.0 / np.pi) * np.imag(g_total)

    record_keeping = {}
    record_keeping.update(krylov_dim_record)
    record_keeping.update(krylov_zero_record)
    record_keeping.update(gf_record)

    return {
        "g_total": g_total,
        "spectral": spectral,
        "phi_minus_norm": qiskit_phi_minus_norm,
        "phi_plus_norm": qiskit_phi_plus_norm,
        "phi_minus_array": qiskit_phi_minus_array,
        "phi_plus_array": qiskit_phi_plus_array,
        "krylov_dim_minus": krylov_dim_minus,
        "krylov_dim_plus": krylov_dim_plus,
        "sectors": {
            "charge_minus": charge_minus, "spin_minus": spin_minus,
            "charge_plus": charge_plus, "spin_plus": spin_plus,
        },
        "record": record_keeping,
    }


# ============================================================
# E. calculate_relative_errors()
# Direct, verified port of dmft.calculate_rel_errors() -- confirmed
# against dmft.py's actual source line for line. Same formulas, same
# record_keeping keys (g_numerator, g_denominator, g_rel_error,
# g_avg_rel_diff), just renamed g_vqe -> g_qiskit.
# ============================================================

def calculate_relative_errors(g_qiskit, g_exact):
    """
    Calculate the relative error between the Green's function found
    exactly and the one found by Qiskit VQE.

    :param g_qiskit: Green's function generated by Qiskit VQE.
    :param g_exact: Green's function generated by exact solution.
    :return: (rel_error, record_keeping)
    """
    avg_rel_diff = 0
    rel_diffs = []
    for z, gz_qiskit in enumerate(g_qiskit):
        diff = np.sqrt(
            (gz_qiskit.real - g_exact[z].real) ** 2
            + (gz_qiskit.imag - g_exact[z].imag) ** 2
        )
        avg_rel_diff += diff
        rel_diffs.append(diff)
    avg_rel_diff /= len(g_qiskit)

    numerator = np.linalg.norm(np.subtract(g_qiskit, g_exact))
    denominator = np.linalg.norm(g_exact)
    rel_error = numerator / denominator

    record_keeping = {
        "g_numerator": numerator,
        "g_denominator": denominator,
        "g_rel_error": rel_error,
        "g_avg_rel_diff": avg_rel_diff,
        "g_max_difference": float(np.max(rel_diffs)) if rel_diffs else 0.0,
    }

    return rel_error, record_keeping


# ============================================================
# G. Plotting
# Split into two focused functions (the original plot_gfs() draws
# both real and spectral parts on one axes) -- same data, same
# real/spectral formulas (-Im(G)/pi), just separated per the plan.
# ============================================================

def plot_green_functions(w, g_qiskit, g_exact, impurity_orbital=None):
    """Plot the real part of G(omega) for both methods."""
    omega = w.real
    plt.figure(figsize=(9, 5.5))
    plt.plot(omega, g_qiskit.real, "b-", label="Qiskit Re[G]")
    plt.plot(omega, g_exact.real, color="tab:blue", linestyle="--", label="Exact Re[G]")
    plt.legend()
    plt.xlabel(r"$\omega$")
    label = r"$G^{ret}_%s(\omega)$" % impurity_orbital if impurity_orbital is not None else r"$G^{ret}(\omega)$"
    plt.ylabel(label)
    plt.title("Qiskit vs Exact Green's Function (real part)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_spectral_functions(w, spectral_qiskit, spectral_exact):
    """Plot the spectral function A(omega) = -Im(G)/pi for both methods."""
    omega = w.real
    plt.figure(figsize=(9, 5.5))
    plt.plot(omega, spectral_qiskit, "g-", label="Qiskit spectral")
    plt.plot(omega, spectral_exact, color="tab:green", linestyle="--", label="Exact spectral")
    plt.legend()
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$A(\omega)$")
    plt.title("Qiskit vs Exact Spectral Function")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# H. save_results() -- STUB, deliberately deferred.
#
# The original dmft.py has JSON/PKL saving plus an experiment-directory
# structure (save_to_file()). Per the build plan, this is intentionally
# NOT implemented yet -- there's no point designing a results-storage
# format around a pipeline that isn't finished end to end. Add this
# once run_aim_qiskit() is validated.
# ============================================================

def save_results(results, filename):
    raise NotImplementedError(
        "save_results() is deliberately not implemented yet -- see the "
        "module docstring / build plan. Add this once run_aim_qiskit() "
        "is validated end to end."
    )


# ============================================================
# F. run_aim_qiskit()
# Fixed-depth end-to-end driver. For original-style automatic depth
# selection, use run_gs_error_experiment_qiskit() below.
# ============================================================

def run_aim_qiskit(
    system_size,
    seed=0,
    vqe_depth=1,
    optimizer="BFGS",
    gs_gtol=5e-4,
    gf_gtol=5e-5,
    conv_tol=1e-6,
    maxiter=1_000_000,
    w=None,
    calculate_exact=True,
    display=True,
    plot=False,
    optimizer_seed=0,
    seed_minus=0,
    seed_plus=0,
):
    """Run one fixed-depth Qiskit AIM calculation end to end."""
    if w is None:
        w = np.linspace(
            -25,
            25,
            1000,
            dtype=np.complex128,
        ) + 0.1j

    (
        impurity_orbital,
        test_model,
        n_orbitals,
        up_qubit_indices,
        down_qubit_indices,
        connected_graphs,
        qubit_hamiltonian,
    ) = initialize_system(system_size, seed)

    results = {
        "system": {
            "system_size": system_size,
            "seed": seed,
            "impurity_orbital": impurity_orbital,
            "n_orbitals": n_orbitals,
            "up_qubit_indices": up_qubit_indices,
            "down_qubit_indices": down_qubit_indices,
        }
    }

    if display:
        print(
            f"System initialized: N={system_size}, seed={seed}, "
            f"impurity_orbital={impurity_orbital}"
        )

    exact_energy = None
    exact_state = None
    exact_charge = None
    exact_spin = None
    exact_gs_record = None
    degenerate = False

    if calculate_exact:
        (
            exact_energy,
            exact_state,
            exact_charge,
            exact_spin,
            degenerate,
            exact_gs_record,
        ) = exact.solve_exact_gs(test_model)

        results["exact_ground_state"] = {
            "energy": exact_energy,
            "charge": exact_charge,
            "spin": exact_spin,
            "degenerate": degenerate,
            "record": exact_gs_record,
        }

        if display:
            print(
                f"Exact GS: energy={exact_energy:.12f}, "
                f"charge={exact_charge}, spin={exact_spin}, "
                f"degenerate={degenerate}"
            )

    np.random.seed(optimizer_seed)

    gs_result = calculate_ground_state_qiskit(
        test_model=test_model,
        qubit_hamiltonian=qubit_hamiltonian,
        connected_graphs=connected_graphs,
        up_qubit_indices=up_qubit_indices,
        down_qubit_indices=down_qubit_indices,
        vqe_depth=vqe_depth,
        optimizer=optimizer,
        gs_gtol=gs_gtol,
        maxiter=maxiter,
        display=display,
    )
    results["ground_state"] = gs_result

    if calculate_exact and not degenerate:
        gs_compare = compare_ground_states_qiskit(
            exact_energy=exact_energy,
            qiskit_energy=gs_result["energy"],
            exact_state=exact_state,
            qiskit_state_ordered=gs_result["state_ordered"],
        )
        results["ground_state"]["exact"] = {
            "energy": exact_energy,
            "charge": exact_charge,
            "spin": exact_spin,
        }
        results["ground_state"]["comparison"] = gs_compare

    qiskit_gf = calculate_green_function_qiskit(
        impurity_orbital=impurity_orbital,
        qubit_hamiltonian=qubit_hamiltonian,
        ground_state_result=gs_result,
        up_qubit_indices=up_qubit_indices,
        down_qubit_indices=down_qubit_indices,
        connected_graphs=connected_graphs,
        n_orbitals=n_orbitals,
        vqe_depth=vqe_depth,
        w=w,
        optimizer=optimizer,
        conv_tol=conv_tol,
        maxiter=maxiter,
        gf_gtol=gf_gtol,
        display=display,
        seed_minus=seed_minus,
        seed_plus=seed_plus,
    )
    results["green_function"] = qiskit_gf

    # Exact GF is comparison-only and must not run when exact was disabled
    # or the exact ground state is degenerate.
    if calculate_exact and not degenerate:
        new_phi_plus_ordered = qulacs_to_python_ordering_qiskit(
            qiskit_gf["phi_plus_array"],
            gs_result["n_qubits"],
        )
        new_phi_minus_ordered = qulacs_to_python_ordering_qiskit(
            qiskit_gf["phi_minus_array"],
            gs_result["n_qubits"],
        )

        g_exact, exact_gf_record = exact.calculate_gf_exact(
            exact_gs=exact_state,
            test_model=test_model,
            new_phi_plus=new_phi_plus_ordered,
            new_phi_minus=new_phi_minus_ordered,
            n_orbitals=n_orbitals,
            w=w,
            exact_gs_energy=exact_energy,
            gs=False,
            vqe_krylov_ideal_dim_plus=qiskit_gf["krylov_dim_plus"],
            vqe_krylov_ideal_dim_minus=qiskit_gf["krylov_dim_minus"],
            impurity_orbital=impurity_orbital,
        )
        spectral_exact = -(1.0 / np.pi) * np.imag(g_exact)

        results["green_function"]["exact"] = {
            "g_total": g_exact,
            "spectral": spectral_exact,
            "record": exact_gf_record,
        }

        rel_error, rel_error_record = calculate_relative_errors(
            qiskit_gf["g_total"],
            g_exact,
        )
        results["errors"] = rel_error_record

        if display:
            print(
                "Green's function relative error "
                f"(Qiskit vs exact): {rel_error:.6e}"
            )

        if plot:
            plot_gfs_qiskit(
                w=w,
                g_qiskit=qiskit_gf["g_total"],
                g_exact=g_exact,
                impurity_orbital=impurity_orbital,
                n_layers=vqe_depth,
                g_rel_error=rel_error,
            )

    elif calculate_exact and degenerate and display:
        print(
            "Exact ground state is degenerate; "
            "Qiskit GF was computed but exact comparison was skipped."
        )

    return results

def run_gs_error_experiment_qiskit(
    system_size,
    target_err,
    seed=0,
    gf_maxiter=1_000_000,
    gs_maxiter=1_000_000_000,
    gs_gtol=5e-4,
    gf_gtol=5e-5,
    pre_empt_layers=4,
    starting_depth=1,
    gs=False,
    display=True,
    plot=False,
    optimizer="BFGS",
    conv_tol=1e-6,
    w=None,
    optimizer_seed=0,
):
    """
    Qiskit equivalent of dmft.run_gs_error_experiment().

    Increase VQE depth from starting_depth through pre_empt_layers.
    Stop when the ground-state overlap error falls below target_err.
    If that ground-state optimization also reports success and gs=False,
    calculate the Green's function once at the accepted depth.
    """

    if starting_depth > pre_empt_layers:
        raise ValueError(
            "starting_depth must not be greater than "
            "pre_empt_layers."
        )

    if w is None:
        w = (
            np.linspace(
                -25,
                25,
                1000,
                dtype=np.complex128,
            )
            + 0.1j
        )

    (
        impurity_orbital,
        test_model,
        n_orbitals,
        up_qubit_indices,
        down_qubit_indices,
        connected_graphs,
        qubit_hamiltonian,
    ) = initialize_system(
        system_size,
        seed,
    )

    (
        exact_energy,
        exact_state,
        exact_charge,
        exact_spin,
        degenerate,
        exact_gs_record,
    ) = exact.solve_exact_gs(
        test_model
    )

    results = {
        "system": {
            "system_size": system_size,
            "seed": seed,
            "impurity_orbital": impurity_orbital,
            "n_orbitals": n_orbitals,
            "up_qubit_indices": up_qubit_indices,
            "down_qubit_indices": down_qubit_indices,
        },
        "exact_ground_state": {
            "energy": exact_energy,
            "charge": exact_charge,
            "spin": exact_spin,
            "degenerate": degenerate,
            "record": exact_gs_record,
        },
        "depth_history": {},
        "selected_depth": None,
        "overall_success": False,
    }

    if degenerate:
        if display:
            print(
                "Exact ground state is degenerate. "
                "Depth-scaling run stopped."
            )
        return results

    # Deterministic optimization sequence.
    np.random.seed(
        optimizer_seed
    )

    selected_gs = None

    for n_layers in range(
        starting_depth,
        pre_empt_layers + 1,
    ):

        if display:
            print()
            print("=" * 72)
            print(
                f"N={system_size}, "
                f"L={n_layers}, "
                f"GS gtol={gs_gtol:.1e}"
            )
            print("=" * 72)

        gs_result = calculate_ground_state_qiskit(
            test_model=test_model,
            qubit_hamiltonian=qubit_hamiltonian,
            connected_graphs=connected_graphs,
            up_qubit_indices=up_qubit_indices,
            down_qubit_indices=down_qubit_indices,
            vqe_depth=n_layers,
            optimizer=optimizer,
            gs_gtol=gs_gtol,
            maxiter=gs_maxiter,
            display=display,
        )

        gs_compare = compare_ground_states_qiskit(
            exact_energy=exact_energy,
            qiskit_energy=gs_result["energy"],
            exact_state=exact_state,
            qiskit_state_ordered=(
                gs_result["state_ordered"]
            ),
        )

        gs_result["exact"] = {
            "energy": exact_energy,
            "charge": exact_charge,
            "spin": exact_spin,
        }

        gs_result["comparison"] = (
            gs_compare
        )

        local_success = bool(
            gs_result["record"][
                "local_vqe_success"
            ]
        )

        gs_error = float(
            gs_compare["gs_error"]
        )

        results["depth_history"][
            n_layers
        ] = {
            "gs_error": gs_error,
            "gs_energy_rel_error": float(
                gs_compare[
                    "gs_energy_rel_error"
                ]
            ),
            "local_vqe_success": (
                local_success
            ),
            "energy": float(
                gs_result["energy"]
            ),
        }

        if display:
            print(
                f"L={n_layers}: "
                f"GS overlap error={gs_error:.6e}, "
                f"success={local_success}"
            )

        # Match the original workflow:
        # stop when the GS error threshold has been reached.
        if gs_error < target_err:

            results["selected_depth"] = (
                n_layers
            )

            selected_gs = gs_result

            results["overall_success"] = (
                local_success
            )

            break

    if selected_gs is None:

        if display:
            print()
            print(
                "Ground-state target was not reached "
                f"through L={pre_empt_layers}."
            )

        return results

    results["ground_state"] = (
        selected_gs
    )

    if not results["overall_success"]:

        if display:
            print(
                "Ground-state target was reached, "
                "but optimizer success=False. "
                "Green's function will not be run."
            )

        return results

    if gs:
        return results

    selected_depth = results[
        "selected_depth"
    ]

    if display:
        print()
        print("=" * 72)
        print(
            f"GROUND STATE ACCEPTED AT L="
            f"{selected_depth}"
        )
        print("Starting Green's function")
        print("=" * 72)

    qiskit_gf = calculate_green_function_qiskit(
        impurity_orbital=impurity_orbital,
        qubit_hamiltonian=qubit_hamiltonian,
        ground_state_result=selected_gs,
        up_qubit_indices=up_qubit_indices,
        down_qubit_indices=down_qubit_indices,
        connected_graphs=connected_graphs,
        n_orbitals=n_orbitals,
        vqe_depth=selected_depth,
        w=w,
        optimizer=optimizer,
        conv_tol=conv_tol,
        maxiter=gf_maxiter,
        gf_gtol=gf_gtol,
        display=display,
        seed_minus=optimizer_seed,
        seed_plus=optimizer_seed,
    )

    results["green_function"] = (
        qiskit_gf
    )

    new_phi_plus_ordered = (
        qulacs_to_python_ordering_qiskit(
            qiskit_gf[
                "phi_plus_array"
            ],
            selected_gs["n_qubits"],
        )
    )

    new_phi_minus_ordered = (
        qulacs_to_python_ordering_qiskit(
            qiskit_gf[
                "phi_minus_array"
            ],
            selected_gs["n_qubits"],
        )
    )

    (
        g_exact,
        exact_gf_record,
    ) = exact.calculate_gf_exact(
        exact_gs=exact_state,
        test_model=test_model,
        new_phi_plus=(
            new_phi_plus_ordered
        ),
        new_phi_minus=(
            new_phi_minus_ordered
        ),
        n_orbitals=n_orbitals,
        w=w,
        exact_gs_energy=exact_energy,
        gs=False,
        vqe_krylov_ideal_dim_plus=(
            qiskit_gf[
                "krylov_dim_plus"
            ]
        ),
        vqe_krylov_ideal_dim_minus=(
            qiskit_gf[
                "krylov_dim_minus"
            ]
        ),
        impurity_orbital=(
            impurity_orbital
        ),
    )

    spectral_exact = (
        -(1.0 / np.pi)
        * np.imag(g_exact)
    )

    results["green_function"][
        "exact"
    ] = {
        "g_total": g_exact,
        "spectral": spectral_exact,
        "record": exact_gf_record,
    }

    (
        rel_error,
        rel_error_record,
    ) = calculate_relative_errors(
        qiskit_gf["g_total"],
        g_exact,
    )

    results["errors"] = (
        rel_error_record
    )

    results["errors"][
        "g_rel_error"
    ] = rel_error

    if display:
        print()
        print(
            "GF relative error:",
            rel_error,
        )

    if plot:
        plot_gfs_qiskit(
            w=w,
            g_qiskit=(
                qiskit_gf["g_total"]
            ),
            g_exact=g_exact,
            impurity_orbital=(
                impurity_orbital
            ),
            n_layers=(
                selected_depth
            ),
            g_rel_error=(
                rel_error
            ),
        )

    return results

def plot_gfs_qiskit(
    w,
    g_qiskit,
    g_exact,
    impurity_orbital=None,
    n_layers=None,
    g_rel_error=None,
):
    omega = np.real(w)

    spectral_qiskit = (
        -(1.0 / np.pi)
        * np.imag(g_qiskit)
    )

    spectral_exact = (
        -(1.0 / np.pi)
        * np.imag(g_exact)
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        omega,
        np.real(g_qiskit),
        "b-",
        label="Qiskit real",
    )

    plt.plot(
        omega,
        spectral_qiskit,
        "g-",
        label="Qiskit spectral",
    )

    plt.plot(
        omega,
        np.real(g_exact),
        "#00b2ee", linestyle='dashed',
        label="Exact real",
    )

    plt.plot(
        omega,
        spectral_exact,
        "#66cdaa", linestyle='dashed',
        label="Exact spectral",
    )

    plt.xlabel(
        r"$\omega$"
    )

    if impurity_orbital is None:
        plt.ylabel(
            r"$G^{ret}(\omega)$"
        )
    else:
        plt.ylabel(
            rf"$G^{{ret}}_{{{impurity_orbital}}}(\omega)$"
        )

    title_parts = []

    if n_layers is not None:
        title_parts.append(
            rf"$n_{{layers}}={n_layers}$"
        )

    title_parts.append(
        "BFGS"
    )

    if g_rel_error is not None:
        title_parts.append(
            f"GF rel err={g_rel_error:.3f}"
        )

    plt.title(
        ", ".join(title_parts)
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



# ============================================================
# COMMAND-LINE / NOTEBOOK ENTRY POINT
# Makes dmft_qiskit.py runnable like the original dmft.py
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run the Qiskit AIM workflow with automatic VQE-depth "
            "selection, mirroring the original dmft.py driver."
        )
    )

    parser.add_argument(
        "-size", "--system_size", type=int, required=True,
        help="AIM system size N.",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=0,
        help="AIM Hamiltonian seed.",
    )
    parser.add_argument(
        "-t", "--target_error", type=float, required=True,
        help="Target ground-state overlap error for depth selection.",
    )
    parser.add_argument(
        "-m", "--maxiters", type=int, default=1_000_000,
        help="Maximum GF optimizer iterations/evaluations.",
    )
    parser.add_argument(
        "-gsm", "--gs_maxiters", type=int, default=1_000_000_000,
        help="Maximum ground-state optimizer iterations.",
    )
    parser.add_argument(
        "-g", "--gf_gtol", type=float, default=5e-5,
        help="Green's-function BFGS gradient tolerance.",
    )
    parser.add_argument(
        "-gs_gtol", "--gs_gtol", type=float, default=5e-4,
        help="Ground-state BFGS gradient tolerance.",
    )
    parser.add_argument(
        "-pel", "--pre_empt_layers", type=int, default=4,
        help="Maximum ansatz depth to test.",
    )
    parser.add_argument(
        "-svd", "--starting_vqe_depth", type=int, default=1,
        help="First ansatz depth to test.",
    )
    parser.add_argument(
        "-gs", "--ground_state", action="store_true",
        help="Stop after ground-state depth selection.",
    )
    parser.add_argument(
        "-d", "--display", action="store_true",
        help="Display progress.",
    )
    parser.add_argument(
        "-p", "--plot", action="store_true",
        help="Plot the final Green's function.",
    )

    args = parser.parse_args()

    if args.starting_vqe_depth > args.pre_empt_layers:
        raise ValueError(
            "starting_vqe_depth must not be greater than pre_empt_layers."
        )

    return run_gs_error_experiment_qiskit(
        system_size=args.system_size,
        target_err=args.target_error,
        seed=args.seed,
        gf_maxiter=args.maxiters,
        gs_maxiter=args.gs_maxiters,
        gs_gtol=args.gs_gtol,
        gf_gtol=args.gf_gtol,
        pre_empt_layers=args.pre_empt_layers,
        starting_depth=args.starting_vqe_depth,
        gs=args.ground_state,
        display=args.display,
        plot=args.plot,
        optimizer="BFGS",
        conv_tol=1e-6,
        optimizer_seed=0,
    )


if __name__ == "__main__":
    results = main()
