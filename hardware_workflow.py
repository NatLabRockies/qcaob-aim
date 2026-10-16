#!/usr/bin/env python3
"""
FINAL / CHOSEN AIM -> IBM GREEN'S-FUNCTION WORKFLOW
===================================================

Chosen workflow:
    LOCAL OPTIMIZATION + IBM MEASUREMENT OF GF COEFFICIENTS

Why:
    - More stable than shot-only Krylov optimization.
    - Moderate QPU cost.
    - IBM is used where it is scientifically most useful: measuring
      the expectation values that determine the final GF coefficients.
    - Local optimization remains deterministic/stable enough for the
      variational Krylov basis.

LOCAL STEPS
-----------
1. Ground state:
       Aer finite-shot VQE + SPSA
2. GF / Krylov basis:
       local variational Lanczos optimization
       using BFGS/statevector optimization
3. Export optimized circuits + observables

IBM STEP
--------
IBM does NOT optimize the Krylov states.

IBM measures:
    - removal weight
    - addition weight
    - removal/addition H0
    - removal/addition H20
    - <H> for each optimized Krylov state
    - <H^2> for each optimized Krylov state

The GF is then reconstructed classically with the continued fraction.

KRYLOV COUNT
------------
A D-dimensional sector needs D-1 ADDITIONAL Krylov states because u0
already exists.

Size 2:
    D_minus = 2 -> 1 removal state
    D_plus  = 2 -> 1 addition state

Corrected size 3:
    D_minus = 3 -> 2 removal states
    D_plus  = 9 -> 8 addition states

RECOMMENDED ORDER
-----------------
1. Local GS only:
   python3 hardware_workflow.py \
     --prepare-only \
     --size 2 \
     --depth 1 \
     --shots 5000 \
     --maxiters 500 \
     --vqe-seed 1 \
     --shot-seed 9102

2. Local GF/Krylov optimization only:
   python3 hardware_workflow.py --gf-hardware-prepare

3. Safe check, NO IBM submission:
   python3 hardware_workflow.py --gf-hardware-dry-run

4. REAL IBM submission:
   python3 hardware_workflow.py \
     --gf-hardware-submit \
     --backend ibm_fez \
     --shots 5000

5. Retrieve existing IBM job:
   python3 hardware_workflow.py --gf-hardware-retrieve JOB_ID

6. Plot/reconstruct a specific IBM job:
   python3 hardware_workflow.py --gf-hardware-plot JOB_ID

   Or plot the most recently retrieved job:
   python3 hardware_workflow.py --gf-hardware-plot

MULTIPLE IBM JOBS
-----------------
Each newly submitted IBM job gets its own submission metadata file,
measurement file, results JSON, and plot. The shot count in the title
comes from the selected JOB_ID's metadata, not from the current --shots
default.

LEGACY IBM JOBS
---------------
Jobs submitted before per-job archival can still be retrieved. The
script reconstructs the deterministic PUB-label ordering from the GF
export. Because IBM Runtime metadata may contain internal values that
look like shot counts, legacy jobs MUST be assigned their actual shot
count explicitly once:

    python3 hardware_workflow.py \
      --gf-hardware-retrieve JOB_ID \
      --job-shots N

That exact value is saved in the per-job measurement JSON and is then
used automatically whenever that JOB_ID is plotted.

This retrieval fallback does NOT submit a new QPU job.

KEEP THESE CORE FILES
---------------------
    hardware_workflow.py
    dmft.py
    exact.py
    vqe.py
    qiskit_port/

CURRENT RUN OUTPUTS TO KEEP
---------------------------
    aim_hw_data.json
    aim_hw_circuit.qasm
    aim_hw_gf_data.json
    aim_hw_gf_circuits/
    aim_hw_gf_submission.json
    aim_hw_gf_measurements.json
    aim_hw_gf_results.json

Old shot-only experiment helper files are not part of this workflow.
Use cleanup_old_workflows.sh if you want to remove those known files.

Environment split:
    Local:
      /Library/Frameworks/Python.framework/Versions/3.11/bin/python3

    IBM:
      ~/venvs/qcaob-ibm-hardware/bin/python
"""


import argparse
import json
import os
import subprocess
from pathlib import Path

WORKFLOW_BUILD = "job-shot-label-fix-v4"


# ---------------------------------------------------------------------
# Environment / paths
# ---------------------------------------------------------------------

LOCAL_PYTHON = "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
IBM_PYTHON = os.path.expanduser("~/venvs/qcaob-ibm-hardware/bin/python")

SCRIPT = str(Path(__file__).resolve())

DATA_FILE = "aim_hw_data.json"
QASM_FILE = "aim_hw_circuit.qasm"
GS_SUBMISSION_FILE = "aim_hw_submission.json"

GF_DATA_FILE = "aim_hw_gf_data.json"
GF_SUBMISSION_FILE = "aim_hw_gf_submission.json"
GF_MEASUREMENTS_FILE = "aim_hw_gf_measurements.json"
GF_RESULTS_FILE = "aim_hw_gf_results.json"
GF_QASM_DIR = Path("aim_hw_gf_circuits")


def _safe_job_tag(job_id):
    """Filesystem-safe IBM job identifier."""
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(job_id))


def _gf_submission_path(job_id):
    return Path(
        f"aim_hw_gf_submission_{_safe_job_tag(job_id)}.json"
    )


def _gf_measurements_path(job_id):
    return Path(
        f"aim_hw_gf_measurements_{_safe_job_tag(job_id)}.json"
    )


def _gf_results_path(job_id):
    return Path(
        f"aim_hw_gf_results_{_safe_job_tag(job_id)}.json"
    )


def _gf_plot_path(job_id, shots):
    shot_tag = str(int(shots)) if shots is not None else "unknown"
    return Path(
        f"ibm_gf_combined_{_safe_job_tag(job_id)}_{shot_tag}shots.png"
    )


def _recursive_find_shots(obj):
    """Best-effort recovery of shots from IBM Runtime job metadata."""
    if obj is None:
        return None

    if isinstance(obj, dict):
        for key in ("default_shots", "shots"):
            if key in obj:
                try:
                    value = int(obj[key])
                    if value > 0:
                        return value
                except (TypeError, ValueError):
                    pass
        for value in obj.values():
            found = _recursive_find_shots(value)
            if found is not None:
                return found
        return None

    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = _recursive_find_shots(value)
            if found is not None:
                return found
        return None

    for attr in ("default_shots", "shots"):
        try:
            value = getattr(obj, attr)
        except Exception:
            continue
        try:
            value = int(value)
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass

    return None


def _shots_from_runtime_job(job):
    """Return the job's IBM Runtime shot count when the API exposes it."""
    candidates = []

    for attr in ("inputs", "_inputs"):
        try:
            value = getattr(job, attr)
            if callable(value):
                value = value()
            candidates.append(value)
        except Exception:
            pass

    try:
        candidates.append(job.metrics())
    except Exception:
        pass

    for candidate in candidates:
        found = _recursive_find_shots(candidate)
        if found is not None:
            return found

    return None


def _run(interpreter, args):
    cmd = [interpreter, SCRIPT, *args]
    print("\n$", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _serialize_sparse_pauli(op):
    terms = []
    for label, coefficient in op.to_list():
        coefficient = complex(coefficient)
        terms.append(
            {
                "label": str(label),
                "real": float(coefficient.real),
                "imag": float(coefficient.imag),
            }
        )
    return terms


def _deserialize_sparse_pauli(terms):
    from qiskit.quantum_info import SparsePauliOp

    return SparsePauliOp.from_list(
        [
            (
                term["label"],
                complex(term["real"], term["imag"]),
            )
            for term in terms
        ]
    )


def _labels_from_current_gf_export():
    """
    Reconstruct the deterministic PUB-label order used by
    _ibm_gf_submit(). This is a backward-compatibility path for IBM jobs
    submitted before per-job submission metadata was archived.

    Safety rule: retrieval still checks that the number of IBM PUB
    results exactly matches this reconstructed label list.
    """
    if not Path(GF_DATA_FILE).exists():
        raise FileNotFoundError(
            f"{GF_DATA_FILE} is required to reconstruct labels for a "
            "legacy IBM GF job."
        )

    with open(GF_DATA_FILE, "r") as f:
        data = json.load(f)

    labels = [
        "gs.removal_weight",
        "gs.addition_weight",
        "gs.removal_H0",
        "gs.removal_H20",
        "gs.addition_H0",
        "gs.addition_H20",
    ]

    for branch in ("minus", "plus"):
        for state in data["branches"][branch]["states"]:
            iteration = int(state["iteration"])
            labels.append(
                f"{branch}.u{iteration}.H"
            )
            labels.append(
                f"{branch}.u{iteration}.H2"
            )

    return labels


def _recursive_find_precision(obj):
    """Find target precision in IBM Runtime result/job metadata."""
    if obj is None:
        return None

    if isinstance(obj, dict):
        for key in (
            "target_precision",
            "precision",
        ):
            if key in obj:
                try:
                    value = float(obj[key])
                    if value > 0:
                        return value
                except (TypeError, ValueError):
                    pass

        for value in obj.values():
            found = _recursive_find_precision(value)
            if found is not None:
                return found
        return None

    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = _recursive_find_precision(value)
            if found is not None:
                return found
        return None

    for attr in (
        "target_precision",
        "precision",
    ):
        try:
            value = float(getattr(obj, attr))
            if value > 0:
                return value
        except Exception:
            pass

    return None


def _shots_from_result_metadata(result):
    """
    Recover shots from Estimator result metadata when present.

    IBM Runtime may report either an explicit shot count or a target
    precision. For the standard Estimator finite-shot relation,
    precision = 1/sqrt(shots), so an integer shot count can be recovered
    from precision when the metadata does not expose shots directly.
    """
    # First look for an explicit shot count.
    for pub_result in result:
        try:
            metadata = pub_result.metadata
        except Exception:
            metadata = None

        found = _recursive_find_shots(metadata)
        if found is not None:
            return found, "ibm_result_metadata"

    # Then try target precision.
    for pub_result in result:
        try:
            metadata = pub_result.metadata
        except Exception:
            metadata = None

        precision = _recursive_find_precision(
            metadata
        )
        if precision is None:
            continue

        estimate = 1.0 / (precision ** 2)
        rounded = int(round(estimate))

        # Only accept a clean integer-like recovery.
        if (
            rounded > 0
            and abs(estimate - rounded)
            <= max(1e-6, 1e-6 * rounded)
        ):
            return (
                rounded,
                "ibm_result_target_precision",
            )

    return None, None


def _job_backend_name(job):
    """Best-effort backend-name recovery for legacy jobs."""
    for attr in (
        "backend_name",
        "backend",
    ):
        try:
            value = getattr(job, attr)
            if callable(value):
                value = value()
        except Exception:
            continue

        if value is None:
            continue

        if isinstance(value, str):
            return value

        try:
            name = value.name
            if callable(name):
                name = name()
            if name:
                return str(name)
        except Exception:
            pass

    return "unknown"


# =====================================================================
# LOCAL GROUND-STATE PREP / EXPORT
# =====================================================================

def _local_prepare(args):
    import numpy as np
    import dmft
    import qiskit_port.qiskit_vqe as qv

    from qiskit_port.qiskit_ansatz import all_symmetry_ansatzae_qiskit
    from qiskit_port.qiskit_utils import (
        create_qiskit_hamiltonian_matrix,
        get_initial_occupations_indices_qiskit,
    )

    if dmft.BACKEND != "qiskit":
        raise RuntimeError(
            "dmft.BACKEND must be 'qiskit' before exporting hardware data."
        )

    (
        impurity_orbital,
        test_model,
        n_orbitals,
        up_qubit_indices,
        down_qubit_indices,
        connected_graphs,
        qubit_hamiltonian,
    ) = dmft.initialize_system(
        system_size=args.size,
        seed=args.seed,
    )

    (
        vqe_energy,
        _,
        _,
        minimum_angles,
        record,
    ) = dmft._solve_vqe_gs_backend(
        test_model=test_model,
        qubit_hamiltonian=qubit_hamiltonian,
        connected_graphs=connected_graphs,
        up_qubit_indices=up_qubit_indices,
        down_qubit_indices=down_qubit_indices,
        vqe_depth=args.depth,
        optimizer="SPSA",
        gs_gtol=args.gs_gtol,
        maxiters=args.maxiters,
        display=True,
        simulation_mode="shots",
        shots=args.shots,
        seed=args.shot_seed,
        vqe_seed=args.vqe_seed,
    )

    vqe_charge = int(record["vqe_charge"])
    vqe_spin = int(record["vqe_spin"])
    vqe_nu = int(record["vqe_nu"])
    vqe_nd = int(record["vqe_nd"])

    _, _, n_qubits = create_qiskit_hamiltonian_matrix(
        qubit_hamiltonian,
        vqe_energy,
    )

    initial_occupations_indices = (
        get_initial_occupations_indices_qiskit(
            up_qubit_indices,
            down_qubit_indices,
            vqe_nu,
            vqe_nd,
        )
    )

    circuit = all_symmetry_ansatzae_qiskit(
        theta=np.asarray(minimum_angles, dtype=float),
        n_qubits=n_qubits,
        n_layers=args.depth,
        initial_occupations_indices=initial_occupations_indices,
        connected_graphs=connected_graphs,
        compilation="generic",
    )

    H_pauli = qv.openfermion_to_qiskit_hamiltonian(
        qubit_hamiltonian,
        n_qubits,
    )

    # For the hardware workflow, do not silently fall back to a
    # statevector reference.  The optimized energy itself came from
    # finite-shot Aer measurements using SPSA.
    reference_energy = float(vqe_energy)

    Path(QASM_FILE).write_text(circuit.qasm())

    data = {
        "system_size": int(args.size),
        "seed": int(args.seed),
        "vqe_depth": int(args.depth),
        "vqe_seed": int(args.vqe_seed),
        "gs_gtol": float(args.gs_gtol),
        "maxiters": int(args.maxiters),
        "n_qubits": int(n_qubits),
        "impurity_orbital": int(impurity_orbital),
        "n_orbitals": int(n_orbitals),
        "vqe_energy": float(vqe_energy),
        "reference_energy": reference_energy,
        "reference_energy_source": "aer_shots",
        "optimization_mode": "shots",
        "optimizer": "SPSA",
        "aer_shots": int(args.shots),
        "shot_seed": int(args.shot_seed),
        "vqe_charge": vqe_charge,
        "vqe_spin": vqe_spin,
        "vqe_nu": vqe_nu,
        "vqe_nd": vqe_nd,
        "minimum_angles": np.asarray(
            minimum_angles,
            dtype=float,
        ).tolist(),
        "hamiltonian": _serialize_sparse_pauli(H_pauli),
    }

    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print("\n" + "=" * 65)
    print("AIM HARDWARE EXPORT COMPLETE")
    print("=" * 65)
    print("System size:", args.size)
    print("Logical qubits:", n_qubits)
    print("VQE depth:", args.depth)
    print("VQE charge:", vqe_charge)
    print("VQE spin:", vqe_spin)
    print("Aer finite-shot VQE energy:", reference_energy)
    print("Aer shots:", args.shots)
    print("Ground-state optimizer: SPSA")
    print("Circuit:", QASM_FILE)
    print("Metadata:", DATA_FILE)


# =====================================================================
# LOCAL GF USING SAVED GS (existing path)
# =====================================================================

def _load_saved_gs_and_system():
    import numpy as np
    import dmft
    import exact

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    system_size = int(data["system_size"])
    seed = int(data["seed"])
    vqe_depth = int(data["vqe_depth"])
    vqe_energy = float(data["vqe_energy"])
    vqe_charge = int(data["vqe_charge"])
    vqe_spin = int(data["vqe_spin"])
    vqe_nu = int(
        data.get(
            "vqe_nu",
            (vqe_charge + vqe_spin) // 2,
        )
    )
    vqe_nd = int(
        data.get(
            "vqe_nd",
            (vqe_charge - vqe_spin) // 2,
        )
    )
    minimum_angles = np.asarray(
        data["minimum_angles"],
        dtype=float,
    )

    (
        impurity_orbital,
        test_model,
        n_orbitals,
        up_qubit_indices,
        down_qubit_indices,
        connected_graphs,
        qubit_hamiltonian,
    ) = dmft.initialize_system(
        system_size=system_size,
        seed=seed,
    )

    (
        exact_gs_energy,
        exact_gs,
        exact_charge,
        exact_spin,
        degenerate,
        _,
    ) = exact.solve_exact_gs(test_model)

    if degenerate:
        raise RuntimeError(
            "Exact ground state is degenerate; GF workflow stopped."
        )

    return {
        "data": data,
        "system_size": system_size,
        "seed": seed,
        "vqe_depth": vqe_depth,
        "vqe_energy": vqe_energy,
        "vqe_charge": vqe_charge,
        "vqe_spin": vqe_spin,
        "vqe_nu": vqe_nu,
        "vqe_nd": vqe_nd,
        "minimum_angles": minimum_angles,
        "impurity_orbital": impurity_orbital,
        "test_model": test_model,
        "n_orbitals": n_orbitals,
        "up_qubit_indices": up_qubit_indices,
        "down_qubit_indices": down_qubit_indices,
        "connected_graphs": connected_graphs,
        "qubit_hamiltonian": qubit_hamiltonian,
        "exact_gs_energy": exact_gs_energy,
        "exact_gs": exact_gs,
        "exact_charge": exact_charge,
        "exact_spin": exact_spin,
    }


def _base_gf_record(ctx, args):
    return {
        "impurity_orbital": ctx["impurity_orbital"],
        "vqe_depth": ctx["vqe_depth"],
        "optimizer": "BFGS",
        "conv_tol": args.conv_tol,
        "maxiters": args.maxiters,
        "gtol": args.gf_gtol,
        "gs_gtol": ctx["data"].get("gs_gtol"),
        "gs": False,
        "checkpoint_file": "",
        "results_file": "",
        "vqe_gs_energy": ctx["vqe_energy"],
        "vqe_charge": ctx["vqe_charge"],
        "vqe_spin": ctx["vqe_spin"],
        "vqe_nu": ctx["vqe_nu"],
        "vqe_nd": ctx["vqe_nd"],
        "exact_gs_energy": float(ctx["exact_gs_energy"]),
        "exact_charge": int(ctx["exact_charge"]),
        "exact_spin": int(ctx["exact_spin"]),
        "degenerate": False,
    }


def _local_gf(args):
    import numpy as np
    import dmft

    ctx = _load_saved_gs_and_system()
    record = _base_gf_record(ctx, args)

    w = (
        np.linspace(
            -25,
            25,
            args.frequency_points,
            dtype=np.complex128,
        )
        + 1j * args.broadening
    )

    print("\n" + "=" * 65)
    print("GREEN'S FUNCTION FROM SAVED GROUND STATE")
    print("=" * 65)
    print("System size:", ctx["system_size"])
    print("Seed:", ctx["seed"])
    print("VQE depth:", ctx["vqe_depth"])
    print("Saved VQE energy:", ctx["vqe_energy"])
    print("Charge:", ctx["vqe_charge"])
    print("Spin:", ctx["vqe_spin"])
    print("n_up:", ctx["vqe_nu"])
    print("n_down:", ctx["vqe_nd"])
    print("Skipping ground-state VQE optimization.")
    print("Starting variational Lanczos / Green's function...")

    record_gf = dmft.calculate_gf(
        impurity_orbital=ctx["impurity_orbital"],
        test_model=ctx["test_model"],
        n_orbitals=ctx["n_orbitals"],
        up_qubit_indices=ctx["up_qubit_indices"],
        down_qubit_indices=ctx["down_qubit_indices"],
        connected_graphs=ctx["connected_graphs"],
        qubit_hamiltonian=ctx["qubit_hamiltonian"],
        vqe_depth=ctx["vqe_depth"],
        w=w,
        optimizer="BFGS",
        gf_gtol=args.gf_gtol,
        vqe_gs_energy=ctx["vqe_energy"],
        vqe_spin=ctx["vqe_spin"],
        vqe_charge=ctx["vqe_charge"],
        exact_gs=ctx["exact_gs"],
        exact_spin=ctx["exact_spin"],
        exact_charge=ctx["exact_charge"],
        minimum_angles=ctx["minimum_angles"],
        record_keeping=record,
        display=False,
        plot=True,
    )

    print("\n" + "=" * 65)
    print("GREEN'S FUNCTION COMPLETE")
    print("=" * 65)

    if "g_rel_error" in record_gf:
        print("GF relative error:", record_gf["g_rel_error"])
    if "g_avg_rel_diff" in record_gf:
        print(
            "GF average relative difference:",
            record_gf["g_avg_rel_diff"],
        )


# =====================================================================
# LOCAL: PREPARE IBM-QPU-EVALUATED GREEN'S FUNCTION
# =====================================================================

def _clean_qubit_operator(op):
    op.compress()
    return op


def _local_gf_hardware_prepare(args):
    """
    1. Reuse saved ground-state parameters.
    2. Run the existing local/statevector variational-Lanczos optimizer.
    3. Export the optimized Krylov circuits.
    4. Export the Pauli observables needed to re-evaluate the Lanczos
       coefficients on IBM hardware.

    PRODUCTION DESIGN:
    The Krylov optimizer remains local (BFGS/statevector). IBM does not
    optimize the states. The a_i, b_i and particle-addition/removal
    weights used in the final GF are reconstructed from IBM QPU
    expectation-value measurements.

    This is intentional: local optimization is more stable and uses
    much less QPU runtime than hardware-side variational optimization.
    """
    import numpy as np
    import dmft
    import vqe
    import qiskit_port.qiskit_vqe as qv

    from openfermion import FermionOperator, jordan_wigner
    from qiskit_port.qiskit_ansatz import all_symmetry_ansatzae_qiskit
    from qiskit_port.qiskit_utils import (
        create_qiskit_hamiltonian_matrix,
        get_initial_occupations_indices_qiskit,
    )

    ctx = _load_saved_gs_and_system()
    record = _base_gf_record(ctx, args)

    w = (
        np.linspace(
            -25,
            25,
            args.frequency_points,
            dtype=np.complex128,
        )
        + 1j * args.broadening
    )

    print("\n" + "=" * 65)
    print("PREPARING IBM GREEN'S-FUNCTION MEASUREMENTS")
    print("=" * 65)
    print("System size:", ctx["system_size"])
    print("VQE depth:", ctx["vqe_depth"])
    print("Running local variational Lanczos once to obtain Krylov angles.")
    print("No QPU job is submitted in this step.")

    record_gf = dmft.calculate_gf(
        impurity_orbital=ctx["impurity_orbital"],
        test_model=ctx["test_model"],
        n_orbitals=ctx["n_orbitals"],
        up_qubit_indices=ctx["up_qubit_indices"],
        down_qubit_indices=ctx["down_qubit_indices"],
        connected_graphs=ctx["connected_graphs"],
        qubit_hamiltonian=ctx["qubit_hamiltonian"],
        vqe_depth=ctx["vqe_depth"],
        w=w,
        optimizer="BFGS",
        gf_gtol=args.gf_gtol,
        vqe_gs_energy=ctx["vqe_energy"],
        vqe_spin=ctx["vqe_spin"],
        vqe_charge=ctx["vqe_charge"],
        exact_gs=ctx["exact_gs"],
        exact_spin=ctx["exact_spin"],
        exact_charge=ctx["exact_charge"],
        minimum_angles=ctx["minimum_angles"],
        record_keeping=record,
        display=False,
        plot=False,
    )

    _, _, n_qubits = create_qiskit_hamiltonian_matrix(
        ctx["qubit_hamiltonian"],
        ctx["vqe_energy"],
    )

    # ---------------------------------------------------------
    # Rebuild the saved GS circuit
    # ---------------------------------------------------------
    gs_initial_occ = get_initial_occupations_indices_qiskit(
        ctx["up_qubit_indices"],
        ctx["down_qubit_indices"],
        ctx["vqe_nu"],
        ctx["vqe_nd"],
    )

    gs_circuit = all_symmetry_ansatzae_qiskit(
        theta=ctx["minimum_angles"],
        n_qubits=n_qubits,
        n_layers=ctx["vqe_depth"],
        initial_occupations_indices=gs_initial_occ,
        connected_graphs=ctx["connected_graphs"],
        compilation="generic",
    )

    GF_QASM_DIR.mkdir(exist_ok=True)
    gs_qasm = GF_QASM_DIR / "gs.qasm"
    gs_qasm.write_text(gs_circuit.qasm())

    # ---------------------------------------------------------
    # Shifted H and H^2 used by Lanczos
    # ---------------------------------------------------------
    H_shift = (
        ctx["qubit_hamiltonian"]
        - float(ctx["vqe_energy"])
    )
    H2_shift = H_shift ** 2

    # ---------------------------------------------------------
    # c and c^dagger for the impurity orbital.
    #
    # Instead of trying to prepare c|GS> nonunitarily on hardware,
    # measure the equivalent ground-state expectation values:
    #
    # ||c|GS>||^2               = <GS|c^dag c|GS>
    # <phi-|H|phi->             = <GS|c^dag H c|GS> / weight
    # <phi-|H^2|phi->           = <GS|c^dag H^2 c|GS> / weight
    #
    # and likewise for c^dag|GS>.
    # ---------------------------------------------------------
    orbital = int(ctx["impurity_orbital"])

    cm = jordan_wigner(
        FermionOperator(
            f"{orbital}",
            1.0,
        )
    )
    cp = jordan_wigner(
        FermionOperator(
            f"{orbital}^",
            1.0,
        )
    )

    observables_of = {
        "H": H_shift,
        "H2": H2_shift,
        "removal_weight": cp * cm,
        "addition_weight": cm * cp,
        "removal_H0": cp * H_shift * cm,
        "removal_H20": cp * H2_shift * cm,
        "addition_H0": cm * H_shift * cp,
        "addition_H20": cm * H2_shift * cp,
    }

    observables = {}
    for name, op in observables_of.items():
        op = _clean_qubit_operator(op)
        pauli = qv.openfermion_to_qiskit_hamiltonian(
            op,
            n_qubits,
        )
        observables[name] = _serialize_sparse_pauli(pauli)

    # ---------------------------------------------------------
    # Determine plus/minus sectors.
    # ---------------------------------------------------------
    (
        _,
        _,
        charge_minus,
        spin_minus,
        ideal_dim_minus,
        _,
        _,
        charge_plus,
        spin_plus,
        ideal_dim_plus,
        _,
    ) = vqe.construct_vqe_krylov_dimensions(
        up_idx=ctx["up_qubit_indices"],
        impurity_orbital=ctx["impurity_orbital"],
        n_orbitals=ctx["n_orbitals"],
        vqe_charge=ctx["vqe_charge"],
        vqe_spin=ctx["vqe_spin"],
    )

    # ---------------------------------------------------------
    # Export locally optimized Krylov ansatz circuits.
    # ---------------------------------------------------------
    branches = {}

    branch_specs = {
        "minus": {
            "charge": int(charge_minus),
            "spin": int(spin_minus),
            "ideal_dim": int(ideal_dim_minus),
            "results_key": "lanczos_iteration_results_minus_vqe",
        },
        "plus": {
            "charge": int(charge_plus),
            "spin": int(spin_plus),
            "ideal_dim": int(ideal_dim_plus),
            "results_key": "lanczos_iteration_results_plus_vqe",
        },
    }

    for branch_name, spec in branch_specs.items():
        n_up = (spec["charge"] + spec["spin"]) // 2
        n_down = (spec["charge"] - spec["spin"]) // 2

        initial_occ = get_initial_occupations_indices_qiskit(
            ctx["up_qubit_indices"],
            ctx["down_qubit_indices"],
            n_up,
            n_down,
        )

        iteration_results = record_gf.get(
            spec["results_key"],
            {},
        )

        exported_states = []

        # Keys are integers in-memory; sort numerically.
        for iteration in sorted(
            iteration_results.keys(),
            key=lambda x: int(x),
        ):
            item = iteration_results[iteration]
            theta = np.asarray(
                item["parameters"],
                dtype=float,
            )

            circuit = all_symmetry_ansatzae_qiskit(
                theta=theta,
                n_qubits=n_qubits,
                n_layers=ctx["vqe_depth"],
                initial_occupations_indices=initial_occ,
                connected_graphs=ctx["connected_graphs"],
                compilation="generic",
            )

            qasm_path = (
                GF_QASM_DIR
                / f"{branch_name}_u{int(iteration)}.qasm"
            )
            qasm_path.write_text(circuit.qasm())

            exported_states.append(
                {
                    "iteration": int(iteration),
                    "qasm": str(qasm_path),
                    "parameters": theta.tolist(),
                    "optimizer_success": bool(
                        item.get("success", False)
                    ),
                    "optimizer_cost": float(
                        item.get("fun", float("nan"))
                    ),
                }
            )

        expected_additional_states = max(
            int(spec["ideal_dim"]) - 1,
            0,
        )

        if len(exported_states) != expected_additional_states:
            raise RuntimeError(
                f"{branch_name}: expected "
                f"{expected_additional_states} additional Krylov "
                f"states for D={spec['ideal_dim']}, but found "
                f"{len(exported_states)}. Check the D-1 fix before "
                "submitting anything to IBM."
            )

        branches[branch_name] = {
            "charge": spec["charge"],
            "spin": spec["spin"],
            "ideal_dim": spec["ideal_dim"],
            "expected_additional_states": expected_additional_states,
            "states": exported_states,
        }

    gf_data = {
        "system_size": int(ctx["system_size"]),
        "seed": int(ctx["seed"]),
        "vqe_depth": int(ctx["vqe_depth"]),
        "n_qubits": int(n_qubits),
        "impurity_orbital": int(ctx["impurity_orbital"]),
        "vqe_energy": float(ctx["vqe_energy"]),
        "vqe_charge": int(ctx["vqe_charge"]),
        "vqe_spin": int(ctx["vqe_spin"]),
        "gs_qasm": str(gs_qasm),
        "frequency_real": [
            float(x) for x in w.real
        ],
        "broadening": float(args.broadening),
        "observables": observables,
        "branches": branches,
        "workflow": "local_optimization_ibm_gf_measurements",
        "krylov_optimizer": "BFGS_local_statevector",
        "ibm_role": "measure_gf_coefficients_only",
        "krylov_count_rule": "D_minus_1_additional_states",

        # Keep local/exact curves for direct comparison later.
        "g_local_real": [
            float(x)
            for x in record_gf.get(
                "g_vqe_real",
                [],
            )
        ],
        "g_local_imag": [
            float(x)
            for x in record_gf.get(
                "g_vqe_imag",
                [],
            )
        ],
        "g_exact_real": [
            float(x)
            for x in record_gf.get(
                "g_exact_real",
                [],
            )
        ],
        "g_exact_imag": [
            float(x)
            for x in record_gf.get(
                "g_exact_imag",
                [],
            )
        ],
        "local_a_minus_real": record_gf.get(
            "a_minus_vqe_real",
            [],
        ),
        "local_b_minus_real": record_gf.get(
            "b_minus_vqe_real",
            [],
        ),
        "local_a_plus_real": record_gf.get(
            "a_plus_vqe_real",
            [],
        ),
        "local_b_plus_real": record_gf.get(
            "b_plus_vqe_real",
            [],
        ),
        "local_gf_relative_error": record_gf.get(
            "g_rel_error"
        ),
        "local_gf_average_relative_difference": (
            record_gf.get("g_avg_rel_diff")
        ),
    }

    with open(GF_DATA_FILE, "w") as f:
        json.dump(gf_data, f, indent=2)

    print("\n" + "=" * 65)
    print("IBM GF EXPORT COMPLETE")
    print("=" * 65)
    print("Metadata:", GF_DATA_FILE)
    print("Circuit directory:", GF_QASM_DIR)
    print(
        "Removal Krylov circuits:",
        len(branches["minus"]["states"]),
    )
    print(
        "Addition Krylov circuits:",
        len(branches["plus"]["states"]),
    )
    print("No IBM job has been submitted yet.")


# =====================================================================
# IBM SERVICE / GROUND-STATE JOBS
# =====================================================================

def _ibm_service():
    import truststore

    truststore.inject_into_ssl()

    from qiskit_ibm_runtime import QiskitRuntimeService

    return QiskitRuntimeService(
        channel="ibm_quantum_platform"
    )


def _load_ibm_export():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit.from_qasm_file(QASM_FILE)

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    H = _deserialize_sparse_pauli(
        data["hamiltonian"]
    )

    return qc, H, data


def _ibm_submit(args):
    from qiskit.transpiler import (
        generate_preset_pass_manager,
    )
    from qiskit_ibm_runtime.executor_estimator import (
        Estimator,
    )

    qc, H, data = _load_ibm_export()

    service = _ibm_service()
    backend = service.backend(args.backend)

    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=3,
        seed_transpiler=args.transpile_seed,
        approximation_degree=1.0,
    )

    isa_circuit = pm.run(qc)
    H_isa = H.apply_layout(isa_circuit.layout)

    estimator = Estimator(mode=backend)
    estimator.options.resilience_level = (
        args.resilience_level
    )
    estimator.options.default_shots = args.shots

    job = estimator.run(
        [
            (
                isa_circuit,
                H_isa,
            )
        ]
    )

    submission = {
        "job_id": job.job_id(),
        "backend": backend.name,
        "shots": int(args.shots),
        "resilience_level": int(
            args.resilience_level
        ),
        "transpile_seed": int(
            args.transpile_seed
        ),
    }

    with open(GS_SUBMISSION_FILE, "w") as f:
        json.dump(submission, f, indent=2)

    print("\n" + "=" * 65)
    print("IBM HARDWARE JOB SUBMITTED")
    print("=" * 65)
    print("System size:", data["system_size"])
    print("Logical qubits:", qc.num_qubits)
    print("Backend:", backend.name)
    print("Shots:", args.shots)
    print("ISA depth:", isa_circuit.depth())
    print("ISA operations:", isa_circuit.count_ops())
    print("Job ID:", job.job_id())
    print("Status:", job.status())


def _ibm_retrieve(args):
    import numpy as np

    service = _ibm_service()

    if not args.job_id:
        raise ValueError(
            "Internal IBM retrieval requires --job-id."
        )

    job = service.job(args.job_id)

    print("Job ID:", args.job_id)
    print("Status:", job.status())

    if "DONE" not in str(job.status()).upper():
        return

    pub_result = job.result()[0]

    energy = float(
        np.asarray(
            pub_result.data.evs
        ).reshape(-1)[0]
    )
    error = float(
        np.asarray(
            pub_result.data.stds
        ).reshape(-1)[0]
    )

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    submission = {}
    if Path(GS_SUBMISSION_FILE).exists():
        with open(GS_SUBMISSION_FILE, "r") as f:
            submission = json.load(f)

    reference = float(data["reference_energy"])
    absolute_difference = abs(
        energy - reference
    )
    relative_difference = (
        absolute_difference / abs(reference)
    )

    # Persist hardware results so later GF plots can show them.
    data["hardware_job_id"] = args.job_id
    data["hardware_backend"] = submission.get(
        "backend"
    )
    data["hardware_shots"] = submission.get(
        "shots"
    )
    data["hardware_energy"] = energy
    data["hardware_uncertainty"] = error
    data[
        "hardware_absolute_difference"
    ] = absolute_difference
    data[
        "hardware_relative_difference"
    ] = relative_difference
    data["hardware_status"] = "DONE"

    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print("\n" + "=" * 65)
    print(
        f"N_sys = {data['system_size']} "
        "IBM HARDWARE GROUND STATE"
    )
    print("=" * 65)
    print(
        "Reference statevector energy:",
        reference,
    )
    print(
        "IBM QPU ground-state energy:",
        energy,
    )
    print("QPU uncertainty:", error)
    print(
        "Absolute difference:",
        absolute_difference,
    )
    print(
        "Relative difference:",
        relative_difference,
    )
    print(
        "Saved hardware measurements to:",
        DATA_FILE,
    )



# =====================================================================
# SAFE GF DRY RUN -- NO IBM SUBMISSION
# =====================================================================

def _gf_hardware_dry_run(args):
    if not Path(GF_DATA_FILE).exists():
        raise FileNotFoundError(
            f"{GF_DATA_FILE} is missing. "
            "Run --gf-hardware-prepare first."
        )

    with open(GF_DATA_FILE, "r") as f:
        data = json.load(f)

    minus_states = data["branches"]["minus"]["states"]
    plus_states = data["branches"]["plus"]["states"]

    minus_D = int(data["branches"]["minus"]["ideal_dim"])
    plus_D = int(data["branches"]["plus"]["ideal_dim"])

    expected_minus = max(minus_D - 1, 0)
    expected_plus = max(plus_D - 1, 0)

    if len(minus_states) != expected_minus:
        raise RuntimeError(
            f"Removal export mismatch: D={minus_D}, expected "
            f"{expected_minus}, found {len(minus_states)}."
        )

    if len(plus_states) != expected_plus:
        raise RuntimeError(
            f"Addition export mismatch: D={plus_D}, expected "
            f"{expected_plus}, found {len(plus_states)}."
        )

    pubs = 6 + 2 * (
        len(minus_states) + len(plus_states)
    )

    print("\\n" + "=" * 72)
    print("LOCAL OPTIMIZATION + IBM GF MEASUREMENT DRY RUN")
    print("=" * 72)
    print("System size:", data["system_size"])
    print("Local Krylov optimizer:", data.get("krylov_optimizer"))
    print("IBM role:", data.get("ibm_role"))
    print(
        f"Removal: D={minus_D}, "
        f"additional states={len(minus_states)}"
    )
    print(
        f"Addition: D={plus_D}, "
        f"additional states={len(plus_states)}"
    )
    print("Estimator PUBs required:", pubs)
    print("IBM backend:", args.backend)
    print("Shots per PUB:", args.shots)
    print("IBM submission: NO")
    print("=" * 72)


# =====================================================================
# IBM QPU-EVALUATED GF: SUBMIT
# =====================================================================

def _ibm_gf_submit(args):
    from qiskit import QuantumCircuit
    from qiskit.transpiler import (
        generate_preset_pass_manager,
    )
    from qiskit_ibm_runtime.executor_estimator import (
        Estimator,
    )

    with open(GF_DATA_FILE, "r") as f:
        data = json.load(f)

    service = _ibm_service()
    backend = service.backend(args.backend)

    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=3,
        seed_transpiler=args.transpile_seed,
        approximation_degree=1.0,
    )

    H = _deserialize_sparse_pauli(
        data["observables"]["H"]
    )
    H2 = _deserialize_sparse_pauli(
        data["observables"]["H2"]
    )

    pubs = []
    labels = []
    circuit_info = []

    # ---------------------------------------------------------
    # Ground-state observables needed for phi_+ / phi_- weights
    # and their a0 / H2 expectation values.
    # ---------------------------------------------------------
    gs = QuantumCircuit.from_qasm_file(
        data["gs_qasm"]
    )
    gs_isa = pm.run(gs)

    gs_observable_names = [
        "removal_weight",
        "addition_weight",
        "removal_H0",
        "removal_H20",
        "addition_H0",
        "addition_H20",
    ]

    for name in gs_observable_names:
        op = _deserialize_sparse_pauli(
            data["observables"][name]
        )
        op_isa = op.apply_layout(
            gs_isa.layout
        )
        pubs.append(
            (
                gs_isa,
                op_isa,
            )
        )
        labels.append(
            f"gs.{name}"
        )

    circuit_info.append(
        {
            "name": "gs",
            "logical_depth": gs.depth(),
            "isa_depth": gs_isa.depth(),
            "isa_ops": {
                str(k): int(v)
                for k, v
                in gs_isa.count_ops().items()
            },
        }
    )

    # ---------------------------------------------------------
    # Each optimized Krylov ansatz state:
    #   <H>  -> a_i
    #   <H^2> -> next b coefficient
    # ---------------------------------------------------------
    for branch in ("minus", "plus"):
        states = data["branches"][branch]["states"]

        for state in states:
            iteration = int(
                state["iteration"]
            )

            qc = QuantumCircuit.from_qasm_file(
                state["qasm"]
            )
            isa = pm.run(qc)

            H_isa = H.apply_layout(
                isa.layout
            )
            H2_isa = H2.apply_layout(
                isa.layout
            )

            pubs.append(
                (
                    isa,
                    H_isa,
                )
            )
            labels.append(
                f"{branch}.u{iteration}.H"
            )

            pubs.append(
                (
                    isa,
                    H2_isa,
                )
            )
            labels.append(
                f"{branch}.u{iteration}.H2"
            )

            circuit_info.append(
                {
                    "name": (
                        f"{branch}.u{iteration}"
                    ),
                    "logical_depth": qc.depth(),
                    "isa_depth": isa.depth(),
                    "isa_ops": {
                        str(k): int(v)
                        for k, v
                        in isa.count_ops().items()
                    },
                }
            )

    estimator = Estimator(mode=backend)
    estimator.options.resilience_level = (
        args.resilience_level
    )
    estimator.options.default_shots = (
        args.shots
    )

    job = estimator.run(pubs)

    submission = {
        "job_id": job.job_id(),
        "backend": backend.name,
        "shots": int(args.shots),
        "resilience_level": int(
            args.resilience_level
        ),
        "transpile_seed": int(
            args.transpile_seed
        ),
        "labels": labels,
        "circuit_info": circuit_info,
    }

    # Save a convenient latest copy AND an immutable per-job copy.
    # This keeps 500-shot, 5,000-shot, 20,000-shot, etc. runs separate.
    with open(GF_SUBMISSION_FILE, "w") as f:
        json.dump(
            submission,
            f,
            indent=2,
        )

    submission_path = _gf_submission_path(
        job.job_id()
    )
    with open(submission_path, "w") as f:
        json.dump(
            submission,
            f,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("IBM GREEN'S-FUNCTION MEASUREMENT JOB SUBMITTED")
    print("=" * 65)
    print("Backend:", backend.name)
    print("Shots per PUB:", args.shots)
    print("Estimator PUBs:", len(pubs))
    print("Job ID:", job.job_id())
    print("Status:", job.status())
    print(
        "Latest submission metadata:",
        GF_SUBMISSION_FILE,
    )
    print(
        "Per-job submission metadata:",
        submission_path,
    )


# =====================================================================
# IBM QPU-EVALUATED GF: RETRIEVE
# =====================================================================

def _ibm_gf_retrieve(args):
    import numpy as np

    if not args.job_id:
        raise ValueError(
            "Internal IBM GF retrieval requires --job-id."
        )

    submission_path = _gf_submission_path(
        args.job_id
    )
    submission = None

    if submission_path.exists():
        with open(submission_path, "r") as f:
            submission = json.load(f)
    elif Path(GF_SUBMISSION_FILE).exists():
        # Backward compatibility: only trust the latest metadata file
        # when it actually belongs to this exact requested job.
        with open(GF_SUBMISSION_FILE, "r") as f:
            candidate = json.load(f)
        if str(candidate.get("job_id")) == str(args.job_id):
            submission = candidate

    service = _ibm_service()
    job = service.job(args.job_id)

    print("GF Job ID:", args.job_id)
    print("Status:", job.status())

    if "DONE" not in str(job.status()).upper():
        return

    result = job.result()

    legacy_reconstructed = False

    if submission is None:
        # This IBM job predates per-job metadata archival. Reconstruct
        # ONLY the deterministic label ordering from the current GF
        # export; never borrow metadata from a different IBM job.
        labels = _labels_from_current_gf_export()

        if len(result) != len(labels):
            raise RuntimeError(
                "Legacy IBM job has "
                f"{len(result)} PUB results, but the current GF export "
                f"implies {len(labels)} labels. Refusing to guess the "
                "mapping. Use the GF export that was used for this job."
            )

        submission = {
            "job_id": args.job_id,
            "backend": _job_backend_name(job),
            "shots": None,
            "resilience_level": None,
            "labels": labels,
            "legacy_reconstructed": True,
        }
        legacy_reconstructed = True

        print(
            "Legacy job: reconstructed PUB labels from "
            f"{GF_DATA_FILE}."
        )
    else:
        labels = submission["labels"]

        if len(result) != len(labels):
            raise RuntimeError(
                "IBM result count does not match the saved "
                "measurement-label count."
            )

    measurements = {}

    for label, pub_result in zip(
        labels,
        result,
    ):
        value = float(
            np.asarray(
                pub_result.data.evs
            ).reshape(-1)[0]
        )
        std = float(
            np.asarray(
                pub_result.data.stds
            ).reshape(-1)[0]
        )

        measurements[label] = {
            "value": value,
            "std": std,
        }

    # Determine the shot count for THIS exact IBM job.
    #
    # IBM Runtime metadata can contain several values that look like a
    # shot count (for example internal execution/twirling metadata).
    # Those values are not reliable enough to label historical plots.
    #
    # NEW JOBS:
    #   Use immutable per-job submission metadata written at submission.
    #
    # LEGACY JOBS:
    #   Require --job-shots N once, using the ACTUAL shot count that
    #   was requested when that exact job was originally submitted.
    #
    # The selected value is saved into the per-job measurement JSON and
    # future plots for that JOB_ID reuse it automatically.
    archived_shots = submission.get("shots")
    override_shots = args.job_shots

    if not legacy_reconstructed and archived_shots is not None:
        job_shots = int(archived_shots)
        shots_source = "per_job_submission_metadata"

        # Runtime values are diagnostic only; never let them overwrite
        # the actual submitted value.
        runtime_shots = _shots_from_runtime_job(job)
        if (
            runtime_shots is not None
            and int(runtime_shots) != job_shots
        ):
            print(
                "NOTE: IBM Runtime exposed a shots-like value of "
                f"{runtime_shots}, but this job's archived submission "
                f"metadata records {job_shots}. Using {job_shots}."
            )

    elif legacy_reconstructed:
        if override_shots is None:
            raise RuntimeError(
                "This is a legacy IBM job and no trustworthy per-job "
                "submission shot count was archived. Re-run retrieval "
                "with --job-shots N, where N is the ACTUAL number of "
                "shots used when this exact job was submitted. "
                "The value will be saved for this job and reused by "
                "future plots. This does NOT submit another QPU job."
            )

        job_shots = int(override_shots)
        if job_shots <= 0:
            raise ValueError(
                "--job-shots must be a positive integer."
            )
        shots_source = "manual_legacy_job_shots"

    elif override_shots is not None:
        job_shots = int(override_shots)
        if job_shots <= 0:
            raise ValueError(
                "--job-shots must be a positive integer."
            )
        shots_source = "manual_job_shots_override"

    else:
        raise RuntimeError(
            "No trustworthy shot count is associated with IBM job "
            f"{args.job_id}. Re-run retrieval with --job-shots N. "
            "This does NOT submit another QPU job."
        )

    output = {
        "job_id": args.job_id,
        "backend": submission.get(
            "backend"
        ),
        "shots": job_shots,
        "shots_source": shots_source,
        "legacy_reconstructed": legacy_reconstructed,
        "resilience_level": submission.get(
            "resilience_level"
        ),
        "measurements": measurements,
    }

    # Latest convenience copy.
    with open(
        GF_MEASUREMENTS_FILE,
        "w",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    # Immutable copy for this IBM job.
    measurements_path = _gf_measurements_path(
        args.job_id
    )
    with open(
        measurements_path,
        "w",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("IBM GF MEASUREMENTS RETRIEVED")
    print("=" * 65)
    print(
        "Measurements:",
        len(measurements),
    )
    print("Shots for this IBM job:", job_shots)
    print("Shot metadata source:", output["shots_source"])
    print("Latest measurements:", GF_MEASUREMENTS_FILE)
    print("Per-job measurements:", measurements_path)


# =====================================================================
# LOCAL: BUILD / PLOT G(w) FROM IBM MEASUREMENTS
# =====================================================================

def _continued_fraction(w, a, b):
    import numpy as np

    w = np.asarray(
        w,
        dtype=np.complex128,
    )
    a = np.asarray(
        a,
        dtype=np.complex128,
    )
    b = np.asarray(
        b,
        dtype=np.complex128,
    )

    if len(a) != len(b):
        raise ValueError(
            "Lanczos a and b arrays must have the same length."
        )

    value = 1.0 / (
        w - a[-1]
    )

    for k in range(
        len(a) - 2,
        -1,
        -1,
    ):
        value = 1.0 / (
            (w - a[k])
            - (b[k + 1] ** 2) * value
        )

    return value


def _local_gf_hardware_plot(args):
    import numpy as np
    import matplotlib.pyplot as plt

    with open(GF_DATA_FILE, "r") as f:
        data = json.load(f)

    requested_job_id = args.job_id

    if requested_job_id:
        measurements_path = _gf_measurements_path(
            requested_job_id
        )
        if not measurements_path.exists():
            raise FileNotFoundError(
                f"{measurements_path} is missing. Retrieve this exact "
                "job first with --gf-hardware-retrieve JOB_ID."
            )
    else:
        measurements_path = Path(
            GF_MEASUREMENTS_FILE
        )

    with open(
        measurements_path,
        "r",
    ) as f:
        hw = json.load(f)

    if (
        requested_job_id
        and str(hw.get("job_id")) != str(requested_job_id)
    ):
        raise RuntimeError(
            "Measurement metadata does not match the requested IBM job."
        )

    measurements = hw["measurements"]

    def value(label):
        return float(
            measurements[label]["value"]
        )

    def build_branch(branch):
        if branch == "minus":
            weight = value(
                "gs.removal_weight"
            )
            h0_num = value(
                "gs.removal_H0"
            )
            h20_num = value(
                "gs.removal_H20"
            )
        else:
            weight = value(
                "gs.addition_weight"
            )
            h0_num = value(
                "gs.addition_H0"
            )
            h20_num = value(
                "gs.addition_H20"
            )

        if np.isclose(weight, 0.0):
            return (
                weight,
                np.array([0.0 + 0.0j]),
                np.array([0.0 + 0.0j]),
                [],
            )

        a = [
            np.complex128(
                h0_num / weight
            )
        ]
        b = [
            np.complex128(0.0)
        ]
        b_squared_raw = []

        states = data[
            "branches"
        ][branch]["states"]

        # Each exported u_i provides a_i.  b_i comes from H^2
        # of the PREVIOUS state.
        for idx, state in enumerate(
            states,
            start=1,
        ):
            iteration = int(
                state["iteration"]
            )

            if idx == 1:
                h2_previous = (
                    h20_num / weight
                )
            else:
                prev_iteration = int(
                    states[idx - 2][
                        "iteration"
                    ]
                )
                h2_previous = value(
                    f"{branch}.u"
                    f"{prev_iteration}.H2"
                )

            b2 = (
                h2_previous
                - a[idx - 1] ** 2
            )

            if idx > 1:
                b2 -= b[idx - 1] ** 2

            b_squared_raw.append(
                complex(b2)
            )
            b.append(
                np.sqrt(
                    np.complex128(b2)
                )
            )

            a_i = value(
                f"{branch}.u"
                f"{iteration}.H"
            )
            a.append(
                np.complex128(a_i)
            )

        return (
            weight,
            np.asarray(
                a,
                dtype=np.complex128,
            ),
            np.asarray(
                b,
                dtype=np.complex128,
            ),
            b_squared_raw,
        )

    (
        weight_minus,
        a_minus,
        b_minus,
        b2_minus_raw,
    ) = build_branch("minus")

    (
        weight_plus,
        a_plus,
        b_plus,
        b2_plus_raw,
    ) = build_branch("plus")

    w = (
        np.asarray(
            data["frequency_real"],
            dtype=np.float64,
        )
        + 1j * float(
            data["broadening"]
        )
    )

    g_minus = (
        weight_minus
        * _continued_fraction(
            -w,
            a_minus,
            b_minus,
        )
    )

    g_plus = (
        weight_plus
        * _continued_fraction(
            w,
            a_plus,
            b_plus,
        )
    )

    g_hw = (
        g_plus
        - g_minus
    )

    g_local = None
    if data.get("g_local_real"):
        g_local = (
            np.asarray(
                data["g_local_real"],
                dtype=float,
            )
            + 1j
            * np.asarray(
                data["g_local_imag"],
                dtype=float,
            )
        )

    g_exact = None
    if data.get("g_exact_real"):
        g_exact = (
            np.asarray(
                data["g_exact_real"],
                dtype=float,
            )
            + 1j
            * np.asarray(
                data["g_exact_imag"],
                dtype=float,
            )
        )

    rel_error_exact = None
    if (
        g_exact is not None
        and np.linalg.norm(g_exact) > 0
    ):
        rel_error_exact = float(
            np.linalg.norm(
                g_hw - g_exact
            )
            / np.linalg.norm(
                g_exact
            )
        )

    rel_error_local = None
    if (
        g_local is not None
        and np.linalg.norm(g_local) > 0
    ):
        rel_error_local = float(
            np.linalg.norm(
                g_hw - g_local
            )
            / np.linalg.norm(
                g_local
            )
        )

    # Ground-state hardware metadata, if a GS job was retrieved.
    gs_data = {}
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, "r") as f:
            gs_data = json.load(f)

    backend = hw.get(
        "backend",
        "IBM QPU",
    )
    # IMPORTANT: the plot shot count belongs to the selected IBM JOB.
    # It is NEVER taken from the current --shots command-line default.
    shots = hw.get("shots")
    job_id = hw.get(
        "job_id",
        "unknown",
    )

    if shots is None:
        raise RuntimeError(
            "Selected IBM job has no reliable shot count metadata. "
            "Refusing to guess the title."
        )

    gs_energy_text = ""
    if (
        gs_data.get("hardware_energy")
        is not None
    ):
        gs_energy_text = (
            "\n"
            f"GS QPU E = "
            f"{gs_data['hardware_energy']:.6f} "
            f"± "
            f"{gs_data.get('hardware_uncertainty', float('nan')):.6f}"
        )

    title_suffix = (
        f"N_sys={data['system_size']}, "
        f"layers={data['vqe_depth']}\n"
        f"{backend}, shots={shots}"
        f"{gs_energy_text}"
    )

    # ---------------------------------------------------------
    # FINAL PRESENTATION PLOT
    #
    # Match the established IBM GF figure format:
    #   solid blue   = IBM real GF
    #   solid orange = IBM spectral function
    #   dashed blue  = exact real GF
    #   dashed orange= exact spectral function
    #
    # Deliberately omit the local-Qiskit curves from this figure.
    # The local result remains available numerically in the JSON output,
    # but the presentation plot compares IBM directly against exact.
    # ---------------------------------------------------------
    A_hw = -np.imag(g_hw) / np.pi
    A_exact = (
        -np.imag(g_exact) / np.pi
        if g_exact is not None
        else None
    )

    fig, ax = plt.subplots(figsize=(20, 9))

    # IBM curves: solid.
    ax.plot(
        w.real,
        np.real(g_hw),
        color="C0",
        linestyle="-",
        linewidth=1.5,
        label="IBM real",
    )
    ax.plot(
        w.real,
        A_hw,
        color="C1",
        linestyle="-",
        linewidth=1.5,
        label="IBM spectral",
    )

    # Exact reference: same colors, dashed.
    if g_exact is not None:
        ax.plot(
            w.real,
            np.real(g_exact),
            color="C0",
            linestyle="--",
            linewidth=1.5,
            label="exact real",
        )
        ax.plot(
            w.real,
            A_exact,
            color="C1",
            linestyle="--",
            linewidth=1.5,
            label="exact spectral",
        )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(
        r"$G^{\mathrm{ret}}(\omega)\;/\;A(\omega)$"
    )

    # Format shots as 5,000 instead of 5000.
    try:
        shots_text = f"{int(shots):,}"
    except (TypeError, ValueError):
        shots_text = str(shots)

    # Match the previous figure title exactly in structure.
    if rel_error_exact is not None:
        error_text = f"{rel_error_exact:.3f}"
    else:
        error_text = "N/A"

    ax.set_title(
        "Qiskit"
        f" | {backend}"
        f" | Shots ({shots_text})"
        f" | $N_{{sys}}$ = {data['system_size']}"
        f" | $n_{{layers}}$ = {data['vqe_depth']}"
        f" | GF Rel. Err.={error_text}"
    )

    print(
        "Plot metadata:"
        f" job_id={job_id},"
        f" shots={shots},"
        f" shots_source={hw.get('shots_source')}"
    )

    ax.set_xlim(
        float(np.min(w.real)),
        float(np.max(w.real)),
    )
    ax.legend(loc="upper right")

    fig.tight_layout()
    plot_path = _gf_plot_path(
        job_id,
        shots,
    )

    # Immutable per-job figure.
    fig.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    # Convenience copy for the most recently plotted job.
    fig.savefig(
        "ibm_gf_combined.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()

    results = {
        "job_id": job_id,
        "backend": backend,
        "shots": shots,
        "weight_minus": float(
            np.real(weight_minus)
        ),
        "weight_plus": float(
            np.real(weight_plus)
        ),
        "a_minus_real": [
            float(x.real)
            for x in a_minus
        ],
        "a_minus_imag": [
            float(x.imag)
            for x in a_minus
        ],
        "b_minus_real": [
            float(x.real)
            for x in b_minus
        ],
        "b_minus_imag": [
            float(x.imag)
            for x in b_minus
        ],
        "b_minus_squared_raw": [
            {
                "real": float(x.real),
                "imag": float(x.imag),
            }
            for x in b2_minus_raw
        ],
        "a_plus_real": [
            float(x.real)
            for x in a_plus
        ],
        "a_plus_imag": [
            float(x.imag)
            for x in a_plus
        ],
        "b_plus_real": [
            float(x.real)
            for x in b_plus
        ],
        "b_plus_imag": [
            float(x.imag)
            for x in b_plus
        ],
        "b_plus_squared_raw": [
            {
                "real": float(x.real),
                "imag": float(x.imag),
            }
            for x in b2_plus_raw
        ],
        "g_ibm_real": [
            float(x.real)
            for x in g_hw
        ],
        "g_ibm_imag": [
            float(x.imag)
            for x in g_hw
        ],
        "relative_error_vs_exact": (
            rel_error_exact
        ),
        "relative_error_vs_local_qiskit": (
            rel_error_local
        ),
    }

    with open(
        GF_RESULTS_FILE,
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    results_path = _gf_results_path(
        job_id
    )
    with open(
        results_path,
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("IBM QPU-EVALUATED GREEN'S FUNCTION COMPLETE")
    print("=" * 65)
    print("Removal weight:", weight_minus)
    print("Addition weight:", weight_plus)
    print("a_minus:", a_minus)
    print("b_minus:", b_minus)
    print("a_plus:", a_plus)
    print("b_plus:", b_plus)

    if rel_error_exact is not None:
        print(
            "GF relative error vs exact:",
            rel_error_exact,
        )

    if rel_error_local is not None:
        print(
            "GF relative error vs local Qiskit:",
            rel_error_local,
        )

    print("Shots used by this IBM job:", shots)
    print("Latest result:", GF_RESULTS_FILE)
    print("Per-job result:", results_path)
    print("Latest plot: ibm_gf_combined.png")
    print("Per-job plot:", plot_path)


# =====================================================================
# CLI
# =====================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Unified AIM ground-state and IBM "
            "Green's-function workflow."
        )
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    # Existing public modes.
    mode.add_argument(
        "--ground-state",
        action="store_true",
        help=(
            "Optimize/export locally, then submit one IBM "
            "ground-state energy job."
        ),
    )
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Optimize/export the ground state locally "
            "without submitting IBM hardware."
        ),
    )
    mode.add_argument(
        "--submit-only",
        action="store_true",
        help=(
            "Submit the currently exported ground-state "
            "circuit without rerunning local VQE."
        ),
    )
    mode.add_argument(
        "--retrieve",
        metavar="JOB_ID",
        help="Retrieve an existing ground-state IBM Runtime job.",
    )
    mode.add_argument(
        "--gf",
        action="store_true",
        help=(
            "Run the existing local GF from saved GS parameters."
        ),
    )

    # New public IBM GF modes.
    mode.add_argument(
        "--gf-hardware-prepare",
        action="store_true",
        help=(
            "Run local variational Lanczos and export the "
            "Krylov circuits/observables needed for IBM GF measurements."
        ),
    )
    mode.add_argument(
        "--gf-hardware-dry-run",
        action="store_true",
        help=(
            "Validate the local GF export and show the IBM PUB count "
            "WITHOUT submitting a QPU job."
        ),
    )
    mode.add_argument(
        "--gf-hardware-submit",
        action="store_true",
        help=(
            "Submit the prepared GF observables/circuits "
            "as one IBM Estimator job."
        ),
    )
    mode.add_argument(
        "--gf-hardware-retrieve",
        metavar="JOB_ID",
        help=(
            "Retrieve an IBM GF measurement job and save "
            "all expectation values."
        ),
    )
    mode.add_argument(
        "--gf-hardware-plot",
        nargs="?",
        const="latest",
        metavar="JOB_ID",
        help=(
            "Build and plot G(w) for one retrieved IBM GF job. "
            "Supply JOB_ID to select a specific run, or omit JOB_ID "
            "to plot the most recently retrieved run."
        ),
    )

    # Hidden worker modes.
    mode.add_argument(
        "--_local-prepare",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    mode.add_argument(
        "--_local-gf",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    mode.add_argument(
        "--_ibm-submit",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    mode.add_argument(
        "--_ibm-retrieve",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    mode.add_argument(
        "--_local-gf-hw-prepare",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    mode.add_argument(
        "--_ibm-gf-submit",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    mode.add_argument(
        "--_ibm-gf-retrieve",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    mode.add_argument(
        "--_local-gf-hw-plot",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--job-id",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--job-shots",
        type=int,
        default=None,
        help=(
            "Legacy retrieval fallback only: manually specify the "
            "shot count for an old IBM job if Runtime metadata cannot "
            "recover it. Does not submit a job."
        ),
    )

    parser.add_argument(
        "--size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--vqe-seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--gs-gtol",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--gf-gtol",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--maxiters",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--shot-seed",
        type=int,
        default=1234,
        help="Random seed for local Aer finite-shot measurements.",
    )
    parser.add_argument(
        "--backend",
        default="ibm_fez",
    )
    parser.add_argument(
        "--resilience-level",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--transpile-seed",
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--conv-tol",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--frequency-points",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--broadening",
        type=float,
        default=0.1,
    )

    return parser


def _prepare_args(args):
    return [
        "--_local-prepare",
        "--size",
        str(args.size),
        "--seed",
        str(args.seed),
        "--depth",
        str(args.depth),
        "--vqe-seed",
        str(args.vqe_seed),
        "--gs-gtol",
        str(args.gs_gtol),
        "--maxiters",
        str(args.maxiters),
        "--shots",
        str(args.shots),
        "--shot-seed",
        str(args.shot_seed),
    ]


def _submit_args(args):
    return [
        "--_ibm-submit",
        "--shots",
        str(args.shots),
        "--backend",
        str(args.backend),
        "--resilience-level",
        str(args.resilience_level),
        "--transpile-seed",
        str(args.transpile_seed),
    ]


def _gf_args(args, internal_mode):
    return [
        internal_mode,
        "--gf-gtol",
        str(args.gf_gtol),
        "--maxiters",
        str(args.maxiters),
        "--conv-tol",
        str(args.conv_tol),
        "--frequency-points",
        str(args.frequency_points),
        "--broadening",
        str(args.broadening),
    ]


def _gf_submit_args(args):
    return [
        "--_ibm-gf-submit",
        "--shots",
        str(args.shots),
        "--backend",
        str(args.backend),
        "--resilience-level",
        str(args.resilience_level),
        "--transpile-seed",
        str(args.transpile_seed),
    ]


def main():
    args = build_parser().parse_args()

    # Visible build identifier so stale copies are obvious.
    if not any(
        [
            args._local_prepare,
            args._local_gf,
            args._ibm_submit,
            args._ibm_retrieve,
            args._local_gf_hw_prepare,
            args._ibm_gf_submit,
            args._ibm_gf_retrieve,
            args._local_gf_hw_plot,
        ]
    ):
        print("Workflow build:", WORKFLOW_BUILD)

    # ---------------------------------------------------------
    # Internal workers
    # ---------------------------------------------------------
    if args._local_prepare:
        _local_prepare(args)
        return

    if args._local_gf:
        _local_gf(args)
        return

    if args._ibm_submit:
        _ibm_submit(args)
        return

    if args._ibm_retrieve:
        _ibm_retrieve(args)
        return

    if args._local_gf_hw_prepare:
        _local_gf_hardware_prepare(args)
        return

    if args._ibm_gf_submit:
        _ibm_gf_submit(args)
        return

    if args._ibm_gf_retrieve:
        _ibm_gf_retrieve(args)
        return

    if args._local_gf_hw_plot:
        _local_gf_hardware_plot(args)
        return

    # ---------------------------------------------------------
    # Public controller commands
    # ---------------------------------------------------------
    if args.ground_state:
        _run(
            LOCAL_PYTHON,
            _prepare_args(args),
        )
        _run(
            IBM_PYTHON,
            _submit_args(args),
        )
        return

    if args.prepare_only:
        _run(
            LOCAL_PYTHON,
            _prepare_args(args),
        )
        return

    if args.submit_only:
        _run(
            IBM_PYTHON,
            _submit_args(args),
        )
        return

    if args.retrieve:
        _run(
            IBM_PYTHON,
            [
                "--_ibm-retrieve",
                "--job-id",
                args.retrieve,
            ],
        )
        return

    if args.gf:
        _run(
            LOCAL_PYTHON,
            _gf_args(
                args,
                "--_local-gf",
            ),
        )
        return

    if args.gf_hardware_prepare:
        _run(
            LOCAL_PYTHON,
            _gf_args(
                args,
                "--_local-gf-hw-prepare",
            ),
        )
        return

    if args.gf_hardware_dry_run:
        _gf_hardware_dry_run(args)
        return

    if args.gf_hardware_submit:
        print()
        print("=" * 72)
        print("WARNING: REAL IBM QPU GF SUBMISSION")
        print("=" * 72)
        print("This consumes IBM QPU runtime.")
        print("If a job ID is returned, DO NOT submit again.")
        print()
        _run(
            IBM_PYTHON,
            _gf_submit_args(args),
        )
        return

    if args.gf_hardware_retrieve:
        retrieve_args = [
            "--_ibm-gf-retrieve",
            "--job-id",
            args.gf_hardware_retrieve,
        ]

        if args.job_shots is not None:
            retrieve_args.extend(
                [
                    "--job-shots",
                    str(args.job_shots),
                ]
            )

        _run(
            IBM_PYTHON,
            retrieve_args,
        )
        return

    if args.gf_hardware_plot:
        plot_args = [
            "--_local-gf-hw-plot",
        ]

        if args.gf_hardware_plot != "latest":
            plot_args.extend(
                [
                    "--job-id",
                    args.gf_hardware_plot,
                ]
            )

        _run(
            LOCAL_PYTHON,
            plot_args,
        )
        return


if __name__ == "__main__":
    main()
