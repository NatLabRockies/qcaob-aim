# Qiskit setup

This document assumes the base repository environment has already been
created using the instructions in the repository's main:

[`README.md`](README.md)

The original project uses Pipenv to manage the Python environment. From
the repository root:

```bash
python3 -m pip install --upgrade pip
pip install virtualenv
pip install pipenv

python3 -m pipenv install
```

Commands in this handoff are written using:

```bash
python3 -m pipenv run <command>
```

so entering an interactive `pipenv shell` is not required.

## Verify the Qiskit dependencies

```bash
python3 -m pipenv run python -c "import numpy, scipy, qiskit, openfermion, dmft, qiskit_port; print('Core imports: OK')"
```

If Qiskit Aer is installed:

```bash
python3 -m pipenv run python -c "import qiskit_aer; print('Qiskit Aer: OK')"
```

Syntax-check the shared and Qiskit-specific source files:

```bash
python3 -m pipenv run python -m py_compile \
    dmft.py \
    qiskit_port/qiskit_utils.py \
    qiskit_port/qiskit_ansatz.py \
    qiskit_port/qiskit_vqe.py \
    qiskit_port/smoke_test_qiskit.py
```

## Select the backend

Qiskit:

```python
import dmft
dmft.BACKEND = "qiskit"
```

Qulacs:

```python
import dmft
dmft.BACKEND = "qulacs"
```

## First Qiskit test

```bash
python3 -m pipenv run python qiskit_port/smoke_test_qiskit.py
```

Expected approximately:

```text
backend          = qiskit
system_size      = 2
seed             = 0
VQE energy       = -5.0639591821482135
GS overlap error ~= 5.64738e-05
optimizer success= True

PASSED
```

---

# AIM Qiskit Port

Qiskit backend for the Anderson Impurity Model (AIM) ground-state VQE and variational-Lanczos Green's-function workflow originally implemented with Qulacs.

This handoff is intended for the next developer maintaining, validating, or extending the port.

The most important architectural point is now:

> **There is one high-level AIM/DMFT workflow in `dmft.py`. Qulacs and Qiskit are selected as backends of that shared workflow.**

The former duplicated high-level file `qiskit_port/dmft_qiskit.py` has been removed.

---

# Start here

If this is your first time opening the project, read:

[`ONBOARDING.md`](ONBOARDING.md)

That document gives the quick-start path into the project. This `README_QISKIT.md` is the full technical reference, including:

- what the AIM workflow is doing,
- repository structure,
- environment setup,
- VQE and ansatz-depth selection,
- variational Lanczos,
- particle-removal and particle-addition branches,
- Green's functions and spectral functions,
- state-ordering rules,
- troubleshooting,
- and a glossary.

Then run the Qiskit smoke test from the repository root:

```bash
python3 -m pipenv run python qiskit_port/smoke_test_qiskit.py
```

After the smoke test passes, open the Qiskit tutorial notebook. In the current repository this is typically:

```text
aim_tutorial_merged.ipynb
```

The tutorial was designed to mirror the original Qulacs tutorial so the two implementations can be followed side by side.

---

# Project status

The Qiskit backend reproduces the tested Qulacs AIM variational workflow.

The validated shared implementation includes:

```text
AIM initialization
        |
        v
exact ground state
        |
        v
shared dmft.py workflow
        |
        v
BACKEND selector
   /          \
Qulacs      Qiskit
   |          |
   v          v
VQE ground-state optimization
        |
        v
automatic ansatz-depth search
        |
        v
particle-removal / particle-addition states
        |
        v
variational Lanczos
        |
        v
continued-fraction Green's function
        |
        v
spectral function
        |
        v
comparison with exact results
```

Both backends have been validated through the same `dmft.py` interface for:

- ground-state optimization,
- automatic ansatz-depth search,
- particle-removal and particle-addition seed states,
- variational Lanczos,
- Green's-function construction,
- exact comparison,
- and result saving.

In controlled same-parameter tests, Qiskit and Qulacs agree to floating-point precision in the ground state, Krylov seed states, Lanczos coefficients, Green's function, and spectral function.

The full debugging history and numerical validation record are in:

[`DEVELOPER_NOTES.md`](DEVELOPER_NOTES.md)


---

# Architecture

The high-level workflow is shared. Only the backend-specific numerical operations remain separate.

```text
                         dmft.py
                            |
                     BACKEND selector
                     /            \
              "qulacs"          "qiskit"
                  |                 |
                vqe.py        qiskit_port/
                                  |
                           qiskit_vqe.py
                                  |
                           qiskit_ansatz.py
                                  |
                           qiskit_utils.py
```

The intended dependency direction is from high-level workflow code toward lower-level backend utilities.

Lower-level Qiskit modules should not import `dmft.py`.

A key maintenance rule is:

> **Do not recreate a separate Qiskit high-level workflow. Add backend dispatch to the shared workflow instead.**

---

# Main source files

## `dmft.py`

Shared high-level AIM/DMFT workflow for both Qulacs and Qiskit.

Main responsibilities include:

- AIM initialization,
- exact ground-state reference calculation,
- backend dispatch,
- VQE ground-state comparison,
- automatic ansatz-depth search,
- Green's-function orchestration,
- error calculation,
- plotting,
- and result saving.

The backend selector is defined near the top of the file:

```python
BACKEND = "qiskit"
# BACKEND = "qulacs"
```

For programmatic use, the selector may also be changed after import and before the workflow is called:

```python
import dmft

dmft.BACKEND = "qiskit"
```

Valid values are:

```text
qiskit
qulacs
```

---

## `qiskit_port/qiskit_utils.py`

Low-level numerical helpers used by the Qiskit backend.

Main responsibilities include:

- expectation values,
- transition amplitudes,
- overlaps,
- continued fractions,
- state-ordering conversion,
- Hamiltonian-matrix construction,
- Hamiltonian-square construction,
- initial occupation indices,
- and cached bit-reversal permutations.

Examples include:

$$
\langle\psi|H|\psi\rangle
$$

and:

$$
\langle\phi|H|\psi\rangle.
$$

This file should remain independent of high-level workflow logic.

---

## `qiskit_port/qiskit_ansatz.py`

Builds the Qiskit symmetry-preserving parameterized circuit.

The ansatz represents a variational state:

$$
|\psi(\theta)\rangle,
$$

where $\theta$ is the vector of trainable circuit parameters.

The graph structure from the original AIM project determines which qubits are connected by the ansatz.

---

## `qiskit_port/qiskit_vqe.py`

Contains the Qiskit-specific variational algorithms.

Main responsibilities include:

- cached parameterized ansatz templates,
- Qiskit/Aer statevector generation,
- VQE ground-state optimization,
- ground-state reconstruction,
- particle-removal and particle-addition seed states,
- variational Lanczos cost evaluation,
- Krylov-state optimization,
- Lanczos coefficient construction,
- removal and addition Green's-function branches,
- and independent branch random seeds.

This is where most backend-specific Qiskit computation lives.

---

## `vqe.py`

Contains the original Qulacs VQE and variational-Lanczos implementation.

The shared `dmft.py` workflow dispatches here when:

```python
BACKEND = "qulacs"
```

Selected backend-independent combinatorial helpers are also reused by the Qiskit path.

The Qulacs implementation remains the primary numerical reference when checking whether the Qiskit backend preserves the original algorithm.

---

## `exact.py`

Provides exact diagonalization and exact Green's-function calculations.

These routines are backend-independent and are shared by both Qulacs and Qiskit.

---

## `n_site_graph_creation.py`

Creates the AIM connectivity graphs used by both backend implementations.

---

# Fast environment check

Run from the repository root.

## Core imports

```bash
python3 -m pipenv run python -c "import numpy, scipy, qiskit, openfermion, dmft, qiskit_port; print('Core imports: OK')"
```

If Qiskit Aer is expected:

```bash
python3 -m pipenv run python -c "import qiskit_aer; print('Qiskit Aer: OK')"
```

## Syntax check

```bash
python3 -m pipenv run python -m py_compile \
    dmft.py \
    qiskit_port/qiskit_utils.py \
    qiskit_port/qiskit_ansatz.py \
    qiskit_port/qiskit_vqe.py \
    qiskit_port/smoke_test_qiskit.py
```

## Known working dependency family

The port was validated in the older Qiskit package family used by the repository, including a working environment with approximately:

```text
NumPy          1.26.4
Qiskit         0.46.3
Qiskit Terra   0.46.3
```

Do not upgrade NumPy to 2.x or migrate to a newer incompatible Qiskit API without rerunning the full regression suite.

For the full environment history, see `DEVELOPER_NOTES.md`.

---

# Smoke test

Run:

```bash
python3 -m pipenv run python qiskit_port/smoke_test_qiskit.py
```

The smoke test explicitly selects:

```python
dmft.BACKEND = "qiskit"
```

and runs a small ground-state calculation through the shared `dmft.py` interface.

It intentionally stops before the more expensive variational-Lanczos Green's-function calculation.

The validated checkpoint is approximately:

```text
backend           = qiskit
system_size       = 2
seed              = 0
VQE energy        = -5.063959182148
GS overlap error  = 5.64738e-05
optimizer success = True
```

The smoke test verifies that:

- the environment imports correctly,
- AIM initialization works,
- exact diagonalization works,
- shared backend dispatch selects Qiskit correctly,
- Qiskit VQE runs,
- state ordering is correct,
- and the ground-state comparison matches the validated reference.

This is a regression/environment test, not a production benchmark.

---

# Recommended production entry point

For normal production-style depth search, use the shared experiment driver.

```python
import dmft

dmft.BACKEND = "qiskit"

assert dmft.BACKEND == "qiskit"

dmft.run_gs_error_experiment(
    system_size=4,
    seed=0,
    target_err=1e-4,
    maxiters=int(1e6),
    gs_gtol=5e-4,
    gf_gtol=5e-5,
    pre_empt_layers=4,
    starting_depth=1,
    gs=False,
    display=True,
    plot=True,
)
```

The high-level sequence is:

```text
L = starting_depth
        |
        v
calculate_gs()
        |
        v
backend dispatch
   /          \
Qulacs      Qiskit
        |
        v
compare with exact GS
        |
        v
GS error < target?
   |          |
  no         yes
   |          |
 L = L+1      v
          calculate_gf()
              |
              v
         backend dispatch
          /          \
       Qulacs       Qiskit
              |
              v
        exact GF comparison
              |
              v
          save results
```

The spectral function is:

$$
A(\omega)
=
-\frac{1}{\pi}\operatorname{Im}G(\omega).
$$

## Return behavior

`run_gs_error_experiment()` preserves the original workflow behavior and returns `None`.

Its results are printed when `display=True` and are saved through the existing result-saving logic.

If a developer needs direct access to returned records for debugging, benchmarking, or tests, use `calculate_gs()` and `calculate_gf()` directly.

---


# Running Qiskit from the terminal

Run commands from the **repository root**, not from inside
`AIM_Qiskit_Handoff/`.

For the current repository layout, first move to the repository root:

```bash
cd /Users/aporter2/Downloads/nlr_qc/qcaob-aim
```

The following command launches Python inside the project's Pipenv
environment, selects the Qiskit backend, and runs the shared automatic
ground-state depth search:

```bash
python3 -m pipenv run python - <<'PY'
import dmft

dmft.BACKEND = "qiskit"

dmft.run_gs_error_experiment(
    system_size=4,
    seed=0,
    target_err=1e-4,
    maxiters=int(1e6),
    gs_gtol=5e-4,
    gf_gtol=5e-5,
    pre_empt_layers=4,
    starting_depth=1,
    gs=True,
    display=True,
    plot=False,
)
PY
```

The `<<'PY' ... PY` syntax is a shell **here-document**. It means that
the Python code between the two `PY` markers is passed directly to the
Python interpreter. There is no separate `.py` file containing this
whole command.

The function being called is:

```python
dmft.run_gs_error_experiment(...)
```

and that function lives in the repository's shared:

```text
dmft.py
```

Because:

```python
dmft.BACKEND = "qiskit"
```

is set before the function call, the shared workflow dispatches the
backend-specific quantum work to the Qiskit implementation.

With:

```python
gs=True
```

the run stops after the ground-state depth search. It does **not** run
the variational-Lanczos Green's-function calculation.

The results are printed directly in the terminal because:

```python
display=True
```

is enabled.

For the validated shared `N=4`, `seed=0`, `target_err=1e-4` case, the
search should progress through the ansatz depths and first satisfy the
target at approximately:

```text
L = 4
```

To run the full ground-state plus Green's-function workflow, use the
same command but change:

```python
gs=False
```

and, if a plot is desired:

```python
plot=True
```

The `AIM_Qiskit_Handoff/` directory contains the documentation for this
workflow. The executable code remains in the repository root and
`qiskit_port/`.

---


# Ground-state-only depth search

During development, it is often better to stop after the ground-state stage.

Use:

```python
import dmft

dmft.BACKEND = "qiskit"

dmft.run_gs_error_experiment(
    system_size=4,
    seed=0,
    target_err=1e-4,
    maxiters=int(1e6),
    gs_gtol=5e-4,
    gf_gtol=5e-5,
    pre_empt_layers=4,
    starting_depth=1,
    gs=True,
    display=True,
    plot=False,
)
```

For the validated shared $N=4$, `seed=0`, `target_err=1e-4` case, the first accepted depth is:

```text
L = 4
```

A representative shared Qiskit depth search produced approximately:

| Depth | GS overlap error |
|---:|---:|
| 1 | $2.145286\times10^{-2}$ |
| 2 | $2.114759\times10^{-4}$ |
| 3 | $1.631186\times10^{-4}$ |
| 4 | $1.004951\times10^{-5}$ |

See `DEVELOPER_NOTES.md` for the full validation history and older controlled reference runs.

---

# Fixed-depth ground-state run

For controlled tests where the ansatz depth must be fixed explicitly, call `calculate_gs()` directly.

Example:

```python
import os
import tempfile
import dmft


dmft.BACKEND = "qiskit"

(
    impurity_orbital,
    test_model,
    n_orbitals,
    up_qubit_indices,
    down_qubit_indices,
    connected_graphs,
    qubit_hamiltonian,
) = dmft.initialize_system(
    system_size=2,
    seed=0,
)

with tempfile.TemporaryDirectory() as tmpdir:
    checkpoint_file = os.path.join(tmpdir, "checkpoint.pkl")
    results_file = os.path.join(tmpdir, "results.pkl")

    record_gs, exact_gs, minimum_angles = dmft.calculate_gs(
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
        display=True,
        gs=True,
        checkpoint_file=checkpoint_file,
        results_file=results_file,
    )
```

Use a fixed-depth run for:

- backend comparisons,
- profiling,
- reproducing a known depth,
- controlled validation,
- and debugging a specific ansatz layer count.

---

# Direct Green's-function debugging

If a validated ground state is already available, `calculate_gf()` can be called directly for controlled testing.

The normal production path should still go through `run_gs_error_experiment(..., gs=False)` so the original acceptance and saving logic remains intact.

---

# Validated shared end-to-end regression

A small $N=2$, `seed=0`, depth-1 end-to-end test has been run through the same `dmft.py` workflow for both backends.

## Qiskit

```text
VQE GS energy     = -5.0639591821482135
GS overlap error  = 5.64737793247e-05
GF relative error = 0.0134313363158
```

## Qulacs

```text
VQE GS energy     = -5.063959182148379
GS overlap error  = 5.64738033157e-05
GF relative error = 0.0134313366052
```

Representative backend differences from this independent optimization run are approximately:

```text
GS energy difference      ~ 1.7e-13
GS error difference       ~ 2.4e-11
phi- norm difference      ~ 1.9e-08
phi+ norm difference      ~ 2.8e-09
GF relative-error diff.   ~ 2.9e-10
```

Small Lanczos-coefficient differences in independent runs are expected because the two optimizers can follow slightly different numerical trajectories.

For stronger backend-equivalence claims, use controlled same-parameter comparisons.

---

# Important conventions

These conventions are part of the validated implementation.

Do not change them casually.

---

## Statevector ordering

Raw circuit states and OpenFermion/Python matrix states do not use the same basis-index convention.

Use:

```python
state_ordered = qulacs_to_python_ordering_qiskit(
    state_raw,
    n_qubits,
)
```

when a Qiskit circuit state must interact with the OpenFermion/Python matrix representation.

Keep Krylov states in their raw circuit ordering during the variational circuit workflow.

Do not store reordered states back into the raw Krylov chain.

---

## Ground-state error definition

The workflow uses:

$$
\mathrm{gs\_overlap}
=
\left|
\langle
\psi_{\mathrm{VQE}}
|
\psi_{\mathrm{exact}}
\rangle
\right|
$$

and:

$$
\mathrm{gs\_error}
=
1-\mathrm{gs\_overlap}.
$$

This is based on the **absolute overlap**, not the squared fidelity.

---

## Particle-removal / particle-addition convention

The validated Green's-function convention is:

```text
removal branch:
    evaluate at -w

addition branch:
    evaluate at +w

total:
    G(w) = G_plus(w) - G_minus(w)
```

Do not apply an additional manual removal sign in notebook or calling code.

---

## Exact-GF state-order boundary

The raw Qiskit `phi_minus` and `phi_plus` arrays are used by the Qiskit variational workflow.

Before those states are passed into the exact/OpenFermion Green's-function calculation, they must be converted into the OpenFermion/Python ordering.

This conversion should happen at the backend boundary exactly once.

---

## Complex Lanczos coefficients

The implementation may encounter a complex quantity before computing a Lanczos $b_i$ coefficient.

Only tiny numerical imaginary noise should be removed.

Do not replace the calculation with:

```python
abs(...)
```

or arbitrary clipping unless the algorithm is intentionally being changed and fully revalidated.

---

## Independent branch seeds

Removal and addition variational-Lanczos branches use separate random seeds.

This prevents optimizer retries in one branch from changing the random initial conditions of the other branch.

---

## Optimizer options

Do not pass BFGS-only options such as `gtol` or `eps` into optimizers that do not support them, such as COBYLA.

Optimizer configuration should remain backend-consistent when making Qiskit/Qulacs comparisons.

---

# Known-good $N=4$ reference

For the primary larger validation system:

```text
system_size = 4
seed = 0
target_err = 1e-4
optimizer = BFGS
starting_depth = 1
pre_empt_layers = 4
```

the exact ground-state energy is approximately:

```text
-6.34249498827644
```

and the shared automatic depth search should accept:

```text
L = 4
```

Controlled same-parameter Qiskit/Qulacs tests produced approximately floating-point-level differences in:

```text
ground-state energy
state fidelity
phi- / phi+ states
Lanczos a coefficients
Lanczos b coefficients
Green's function
spectral function
```

Detailed numerical values are intentionally kept in `DEVELOPER_NOTES.md` so this README remains a front-page reference rather than a full lab notebook.

---

# Performance

The main Qiskit runtime cost is repeated statevector generation during optimization.

The final port includes several optimizations:

- parameterized ansatz-template caching,
- parameter binding instead of rebuilding circuits,
- Qiskit Aer statevector simulation when available,
- cached state-ordering permutations,
- and precomputation of fixed Hamiltonian-vector products inside Lanczos optimization.

Qiskit remains slower than Qulacs for this repeated native-statevector workload.

That is a performance limitation, not a known correctness failure.

Controlled tests showed that the two backends can perform equivalent optimizer work and agree numerically while still having substantially different runtimes.

Benchmark values and profiling history are in `DEVELOPER_NOTES.md`.

---

# Dependencies

The working project depends on:

```text
Python
NumPy
SciPy
Matplotlib
Qiskit
Qiskit Aer        optional but recommended
OpenFermion
NetworkX
Jupyter
```

and the original AIM repository modules.

Preserve a known-working environment when possible.

Before upgrading major packages, especially NumPy, Qiskit, SciPy, or OpenFermion, rerun the documented regression checks.

The project previously encountered compatibility failures after dependency changes, so environment changes should be treated as code changes.

---

# Before modifying the implementation

Read:

[`DEVELOPER_NOTES.md`](DEVELOPER_NOTES.md)

before changing any of the following:

```text
state ordering
ansatz parameter order
charge/spin-sector logic
phi- / phi+ construction
Lanczos cost function
complex b_i handling
removal/addition signs
continued-fraction logic
optimizer seeding
branch retry behavior
backend dispatch
```

After a change, validate from cheapest to most expensive:

```text
1. imports / py_compile
2. qiskit_port/smoke_test_qiskit.py
3. fixed-depth ground-state comparison
4. automatic GS depth search
5. controlled ground-state reconstruction
6. phi- / phi+ comparison
7. fixed-theta Lanczos cost
8. removal/addition coefficient chains
9. direct full Green's function
10. full run_gs_error_experiment(..., gs=False)
11. Qiskit vs Qulacs controlled comparison
```

The detailed expected values are in `DEVELOPER_NOTES.md`.

---

# Migration note: retired `dmft_qiskit.py`

Earlier versions of the Qiskit port used:

```text
qiskit_port/dmft_qiskit.py
```

with functions such as:

```text
run_aim_qiskit(...)
run_gs_error_experiment_qiskit(...)
calculate_ground_state_qiskit(...)
calculate_green_function_qiskit(...)
```

That duplicated the high-level workflow already present in `dmft.py`.

After the backend merge was validated, executable dependencies on the old driver were removed and the file was deleted.

New code should use:

```python
import dmft

dmft.BACKEND = "qiskit"
```

and then call the shared workflow.

Do not reintroduce the retired API unless a deliberate compatibility layer is required and tested.

---

# Handoff file map

```text
README.md
    Top-level handoff index and documentation map.

ONBOARDING.md
    New-developer quick start and reading order.

README_QISKIT.md
    Full current technical guide for the merged Qiskit/Qulacs workflow.

DEVELOPER_NOTES.md
    Development history, debugging record, validation evidence, and regression values.

aim_tutorial_merged.ipynb
    Executable Qiskit tutorial matching the original Qulacs tutorial structure.

dmft.py
    Shared high-level Qulacs/Qiskit AIM workflow.

vqe.py
    Qulacs VQE and variational-Lanczos backend.

qiskit_port/smoke_test_qiskit.py
    Fast Qiskit ground-state environment/regression test through shared dmft.py.

qiskit_port/qiskit_utils.py
    Low-level Qiskit numerical helpers.

qiskit_port/qiskit_ansatz.py
    Qiskit ansatz construction.

qiskit_port/qiskit_vqe.py
    Qiskit VQE and variational-Lanczos backend.
```

---

# Recommended reading order

```text
README.md
    |
    v
ONBOARDING.md
    |
    v
README_QISKIT.md
    |
    v
qiskit_port/smoke_test_qiskit.py
    |
    v
aim_tutorial_merged.ipynb
```

When debugging or modifying internals:

```text
DEVELOPER_NOTES.md
```

The central maintenance rule is:

> **First prove that Qiskit and Qulacs agree at the same parameters. Then investigate differences between independent optimizer runs.**

That distinction prevents optimizer-path differences, ansatz limitations, state-ordering issues, and genuine implementation bugs from being confused with one another.
