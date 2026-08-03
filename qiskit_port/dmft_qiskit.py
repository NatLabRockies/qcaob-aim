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

2. run_aim_qiskit() is a SINGLE-DEPTH run (fixed vqe_depth), not the
   depth-scaling search loop run_gs_error_experiment() implements
   (which reruns calculate_gs at increasing VQE depth until a target
   error is hit). That scaling behavior was intentionally left out of
   this first version, consistent with the two-phase build plan
   (ground state first, then Green's function) -- it can be added
   later as a thin wrapper around run_aim_qiskit() once both phases
   are working, the same way the original layers it on top of
   calculate_gs()/calculate_gf().

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
    optimizer,
    gs_gtol,
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
    vqe_energy, vqe_charge, vqe_spin, minimum_angles, record_keeping = qv.solve_vqe_qiskit(
        test_model=test_model,
        vqe_depth=vqe_depth,
        optimizer=optimizer,
        gs_gtol=gs_gtol,
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
# The main end-to-end driver. Single VQE depth (see module docstring
# for why this doesn't replicate run_gs_error_experiment()'s
# depth-scaling search). Mirrors calculate_gs() + calculate_gf()'s
# actual call sequence, confirmed against dmft.py's source.
# ============================================================

def run_aim_qiskit(
    system_size,
    seed=0,
    vqe_depth=1,
    optimizer="COBYLA",
    gs_gtol=5e-5,
    gf_gtol=5e-5,
    conv_tol=1e-5,
    maxiter=50000,
    w=None,
    calculate_exact=True,
    display=True,
    plot=False,
):
    """
    End-to-end Qiskit AIM run: initialize -> exact ground state ->
    Qiskit VQE ground state -> compare -> exact Green's function ->
    Qiskit Green's function -> compare -> return everything.

    :param system_size: AIM system size (same convention as dmft.py --
        n_bath_n_imp = (system_size - 1, 1)).
    :param w: frequency grid; defaults to the same grid dmft.py uses
        (linspace(-25, 25, 1000) + 0.1j) if not supplied.
    :param calculate_exact: if False, skip the exact-diagonalization
        comparison entirely and only run the Qiskit VQE side (useful
        for larger systems where exact diagonalization becomes
        intractable).
    :return: dict with keys system, ground_state, green_function (each
        possibly containing an "exact" sub-key), and errors.
    """
    if w is None:
        w = np.linspace(-25, 25, 1000, dtype=np.complex128) + 1j * 0.1

    results = {}

    # ---- Initialize ----
    (
        impurity_orbital, test_model, n_orbitals, up_qubit_indices,
        down_qubit_indices, connected_graphs, qubit_hamiltonian,
    ) = initialize_system(system_size, seed)

    results["system"] = {
        "system_size": system_size, "seed": seed,
        "impurity_orbital": impurity_orbital, "n_orbitals": n_orbitals,
        "up_qubit_indices": up_qubit_indices, "down_qubit_indices": down_qubit_indices,
    }

    if display:
        print(f"System initialized: system_size={system_size}, seed={seed}, "
              f"impurity_orbital={impurity_orbital}, n_orbitals={n_orbitals}")

    # ---- Exact ground state (optional) ----
    exact_energy = exact_state = exact_charge = exact_spin = None
    degenerate = False

    if calculate_exact:
        exact_energy, exact_state, exact_charge, exact_spin, degenerate, exact_gs_record = (
            exact.solve_exact_gs(test_model)
        )
        if display:
            print(f"Exact ground state: energy={exact_energy:.6f}, "
                  f"charge={exact_charge}, spin={exact_spin}, degenerate={degenerate}")
        if degenerate:
            print("WARNING: exact ground state is degenerate -- skipping all "
                  "exact-vs-Qiskit comparisons for this system.")

    # ---- Qiskit VQE ground state ----
    gs_result = calculate_ground_state_qiskit(
        test_model=test_model,
        qubit_hamiltonian=qubit_hamiltonian,
        connected_graphs=connected_graphs,
        up_qubit_indices=up_qubit_indices,
        down_qubit_indices=down_qubit_indices,
        vqe_depth=vqe_depth,
        optimizer=optimizer,
        gs_gtol=gs_gtol,
        display=display,
    )
    results["ground_state"] = gs_result

    if display:
        print(f"Qiskit VQE ground state: energy={gs_result['energy']:.6f}, "
              f"charge={gs_result['charge']}, spin={gs_result['spin']}")

    # ---- Compare ground states (only if exact was computed and non-degenerate) ----
    if calculate_exact and not degenerate:
        gs_compare = compare_ground_states_qiskit(
            exact_energy=exact_energy,
            qiskit_energy=gs_result["energy"],
            exact_state=exact_state,
            qiskit_state_ordered=gs_result["state_ordered"],
        )
        results["ground_state"]["exact"] = {
            "energy": exact_energy, "charge": exact_charge, "spin": exact_spin,
        }
        results["ground_state"]["comparison"] = gs_compare
        if display:
            print(f"Ground-state comparison: rel_error={gs_compare['gs_energy_rel_error']:.3e}, "
                  f"fidelity={gs_compare['gs_fidelity']:.6f}")

    # ---- Green's functions (only if ground state comparison makes sense) ----
    if calculate_exact and not degenerate:
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
        )
        results["green_function"] = qiskit_gf

        # Exact Green's function reuses the VQE-derived Krylov
        # dimensions (confirmed against dmft.py's actual source: the
        # **record_keeping unpacking into calculate_gf_exact picks up
        # vqe_krylov_ideal_dim_minus/plus, not exact's own combinatorial
        # ideal dimension) -- same truncation depth on both sides, for
        # a fair comparison.
        #
        # new_phi_plus/new_phi_minus must be in the SAME (OpenFermion)
        # ordering as the exact-diagonalization state, since
        # exact.exact_gf_pm() compares them directly against the
        # ladder-operator-applied exact state -- reorder the raw
        # circuit-ordering Krylov arrays before passing them in, same
        # rule as everywhere else in this project.
        new_phi_plus_ordered = qulacs_to_python_ordering_qiskit(
            qiskit_gf["phi_plus_array"], gs_result["n_qubits"]
        )
        new_phi_minus_ordered = qulacs_to_python_ordering_qiskit(
            qiskit_gf["phi_minus_array"], gs_result["n_qubits"]
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
            "g_total": g_exact, "spectral": spectral_exact, "record": exact_gf_record,
        }

        rel_error, rel_error_record = calculate_relative_errors(qiskit_gf["g_total"], g_exact)
        results["errors"] = rel_error_record
        results["errors"]["g_rel_error"] = rel_error

        if display:
            print(f"Green's function relative error (Qiskit vs exact): {rel_error:.3e}")

        if plot:
            plot_green_functions(w, qiskit_gf["g_total"], g_exact, impurity_orbital)
            plot_spectral_functions(w, qiskit_gf["spectral"], spectral_exact)

    return results
