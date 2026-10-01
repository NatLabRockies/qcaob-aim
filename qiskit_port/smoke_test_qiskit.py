"""
Fast smoke test for the Qiskit backend through the shared dmft.py workflow.
"""

from pathlib import Path
import os
import sys
import tempfile


# ------------------------------------------------------------
# Make repository root importable
# ------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import dmft


# ------------------------------------------------------------
# Force the shared workflow to use Qiskit
# ------------------------------------------------------------

dmft.BACKEND = "qiskit"

assert dmft.BACKEND == "qiskit"


# ------------------------------------------------------------
# Small, fast test case
# ------------------------------------------------------------

SYSTEM_SIZE = 2
SEED = 0

REFERENCE_ENERGY = -5.0639591821482135
REFERENCE_GS_ERROR = 5.6473779325e-05


(
    impurity_orbital,
    test_model,
    n_orbitals,
    up_qubit_indices,
    down_qubit_indices,
    connected_graphs,
    qubit_hamiltonian,
) = dmft.initialize_system(
    system_size=SYSTEM_SIZE,
    seed=SEED,
)


with tempfile.TemporaryDirectory() as tmpdir:

    checkpoint_file = os.path.join(
        tmpdir,
        "checkpoint.pkl",
    )

    results_file = os.path.join(
        tmpdir,
        "results.pkl",
    )

    (
        record_gs,
        exact_gs,
        minimum_angles,
    ) = dmft.calculate_gs(
        impurity_orbital=impurity_orbital,
        test_model=test_model,
        up_qubit_indices=up_qubit_indices,
        down_qubit_indices=down_qubit_indices,
        connected_graphs=connected_graphs,
        qubit_hamiltonian=qubit_hamiltonian,
        vqe_depth=1,
        gs_gtol=5e-4,
        optimizer="BFGS",
        gtol=5e-4,
        maxiters=10000,
        conv_tol=1e-6,
        display=False,
        gs=True,
        checkpoint_file=checkpoint_file,
        results_file=results_file,
    )


# ------------------------------------------------------------
# Validate
# ------------------------------------------------------------

energy = record_gs["vqe_gs_energy"]
gs_error = record_gs["gs_error"]
success = bool(record_gs["local_vqe_success"])


problems = []

if abs(
    energy - REFERENCE_ENERGY
) > 1e-8:
    problems.append(
        f"unexpected VQE energy: {energy}"
    )

if abs(
    gs_error - REFERENCE_GS_ERROR
) > 1e-8:
    problems.append(
        f"unexpected GS error: {gs_error}"
    )

if not success:
    problems.append(
        "local_vqe_success was not True"
    )


print("\n" + "=" * 72)
print("SHARED QISKIT BACKEND SMOKE TEST")
print("=" * 72)

print("backend          =", dmft.BACKEND)
print("system_size      =", SYSTEM_SIZE)
print("seed             =", SEED)
print("VQE energy       =", energy)
print("GS overlap error =", gs_error)
print("optimizer success=", success)


if problems:
    print("\nFAILED")

    for problem in problems:
        print(" -", problem)

    raise SystemExit(1)


print("\nPASSED")
