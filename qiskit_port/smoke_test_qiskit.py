"""Fast ground-state smoke test for the Qiskit AIM port.

Run from the root of the existing AIM repository:

    python smoke_test_qiskit.py

This intentionally stops after a one-layer ground-state calculation.
It does not run the Green's-function / variational-Lanczos workflow.
"""

import math

import qiskit_port.dmft_qiskit as dmft_qiskit


SYSTEM_SIZE = 4
SEED = 0

# Loose target so the test accepts L=1 and exits immediately.
TARGET_ERR = 1.0

REFERENCE_ENERGY = -6.248526936637
REFERENCE_GS_ERROR = 2.1452866e-2


results = dmft_qiskit.run_gs_error_experiment_qiskit(
    system_size=SYSTEM_SIZE,
    seed=SEED,
    target_err=TARGET_ERR,
    gf_maxiter=int(1e6),
    gs_maxiter=int(1e6),
    gf_gtol=5e-5,
    gs_gtol=5e-4,
    pre_empt_layers=1,
    starting_depth=1,
    gs=True,
    display=True,
    plot=False,
    optimizer="BFGS",
    conv_tol=1e-6,
    optimizer_seed=SEED,
)

selected_depth = results.get("selected_depth")
overall_success = bool(results.get("overall_success", False))

gs = results.get("ground_state", {})
energy = gs.get("energy")
comparison = gs.get("comparison", {})
gs_error = comparison.get("gs_error")

print("\n" + "=" * 72)
print("QISKIT AIM SMOKE-TEST SUMMARY")
print("=" * 72)
print("system_size      =", SYSTEM_SIZE)
print("seed             =", SEED)
print("selected_depth   =", selected_depth)
print("overall_success  =", overall_success)
print("VQE energy       =", energy)
print("GS overlap error =", gs_error)

problems = []

if selected_depth != 1:
    problems.append(f"expected selected_depth=1, got {selected_depth}")

if not overall_success:
    problems.append("optimizer did not report overall_success=True")

if energy is None:
    problems.append("ground-state energy was not returned")
else:
    # Loose enough for dependency/platform-level floating-point variation,
    # tight enough to catch an obviously wrong code path.
    if not math.isclose(float(energy), REFERENCE_ENERGY, rel_tol=0.0, abs_tol=5e-3):
        problems.append(
            "energy differs substantially from the validated L=1 reference "
            f"({REFERENCE_ENERGY:.12f})"
        )

if gs_error is None:
    problems.append("ground-state overlap error was not returned")
else:
    if not math.isclose(float(gs_error), REFERENCE_GS_ERROR, rel_tol=0.0, abs_tol=5e-3):
        problems.append(
            "GS overlap error differs substantially from the validated L=1 "
            f"reference ({REFERENCE_GS_ERROR:.8e})"
        )

if problems:
    print("\nSMOKE TEST: CHECK REQUIRED")
    for problem in problems:
        print(" -", problem)
    raise SystemExit(1)

print("\nSMOKE TEST: PASS")
print("The Qiskit ground-state workflow is behaving like the validated reference.")
