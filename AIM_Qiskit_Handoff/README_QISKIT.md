# AIM Qiskit Port

Qiskit implementation of the Anderson Impurity Model (AIM) ground-state VQE and variational-Lanczos Green's-function workflow originally implemented with Qulacs.

This handoff is intended for the next developer maintaining, validating, or extending the port.

---

# Start here

If this is your first time opening the project, read:

[`START_HERE.md`](START_HERE.md)

That document contains the full onboarding guide, including:

- what the AIM workflow is doing,
- repository structure,
- environment setup,
- the ground-state smoke test,
- the meaning of the major parameters,
- VQE and ansatz-depth selection,
- variational Lanczos,
- particle-removal and particle-addition branches,
- Green's functions and spectral functions,
- state-ordering rules,
- troubleshooting,
- and a glossary.

Then run:

```bash
python smoke_test_qiskit.py
```

After the smoke test passes, open:

```text
aim_tutorial_qiskit.ipynb
```

The Qiskit tutorial mirrors the structure of the original Qulacs tutorial so the two implementations can be followed side by side.

---

# Project status

The Qiskit port reproduces the tested Qulacs AIM variational workflow.

The validated implementation includes:

```text
AIM initialization
        |
        v
exact ground state
        |
        v
Qiskit VQE ground state
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
comparison with exact and Qulacs results
```

In controlled tests where Qiskit and Qulacs use the same variational parameters, the two backends agree to floating-point precision in the ground state, Krylov seed states, Lanczos coefficients, Green's function, and spectral function.

The full debugging history and numerical validation record are in:

[`DEVELOPER_NOTES.md`](DEVELOPER_NOTES.md)

---

# Documentation

The handoff documentation has three main pieces.

## `README.md`

This file.

Use it as the project front page and quick navigation reference.

---

## `START_HERE.md`

The detailed onboarding and theory walkthrough.

Read this first if you are new to the code.

It explains what each major operation means and how to run the project safely before launching expensive Green's-function calculations.

---

## `DEVELOPER_NOTES.md`

The combined developer-history and validation record.

It documents the work from May through September 2026, including:

- environment and dependency failures,
- early Qulacs baselines,
- the first Qiskit VQE implementation,
- matrix-inverse vs classical-Lanczos vs variational-Lanczos differences,
- statevector-ordering problems,
- Green's-function sign mistakes,
- optimizer divergence tests,
- addition- and removal-branch audits,
- ansatz-depth limitations,
- complex Lanczos-coefficient handling,
- branch RNG behavior,
- Qiskit performance profiling,
- Aer and circuit-caching improvements,
- final Qiskit/Qulacs controlled validation,
- numerical regression values,
- and the "do not reintroduce these bugs" checklist.

`VALIDATION.md` is retained only as a compatibility pointer to `DEVELOPER_NOTES.md`.

---

# Main source files

The Qiskit implementation is intentionally split into layers:

```text
qiskit_port/
    qiskit_utils.py
          |
          v
    qiskit_ansatz.py
          |
          v
    qiskit_vqe.py
          |
          v
    dmft_qiskit.py
```

The normal direction of dependency is upward through this stack.

---

## `qiskit_utils.py`

Low-level numerical helpers.

Main responsibilities include:

- expectation values,
- transition amplitudes,
- overlaps,
- continued fractions,
- state-ordering conversion,
- Hamiltonian-matrix construction,
- Hamiltonian-square construction,
- initial occupation indices.

Examples of operations handled here include:

$$
\langle\psi|H|\psi\rangle
$$

and:

$$
\langle\phi|H|\psi\rangle.
$$

This file should remain independent of high-level workflow logic.

---

## `qiskit_ansatz.py`

Builds the Qiskit symmetry-preserving parameterized circuit.

The ansatz represents a variational state:

$$
|\psi(\theta)\rangle,
$$

where $\theta$ is the vector of trainable circuit parameters.

The graph structure from the original AIM project determines which qubits are connected by the ansatz.

---

## `qiskit_vqe.py`

Contains the Qiskit variational algorithms.

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
- independent branch random seeds.

This is where most backend-specific scientific computation lives.

---

## `dmft_qiskit.py`

High-level Qiskit workflow driver.

This is normally the main programmatic entry point.

It coordinates:

```text
system initialization
exact reference calculation
Qiskit VQE
ground-state comparison
ansatz-depth selection
Green's-function calculation
error calculation
plotting
returned result dictionaries
```

Backend-independent functionality from the original project is reused where appropriate instead of being duplicated.

---

# Original project files still used

The Qiskit port does not replace every original source file.

Important reused components include:

## `dmft.py`

Provides the original Qulacs workflow and backend-independent AIM initialization used by the port.

It remains useful as the reference implementation for backend comparisons.

---

## `exact.py`

Provides exact diagonalization and exact Green's-function calculations.

These routines are used as the classical reference for manageable system sizes.

---

## `vqe.py`

Contains the original Qulacs VQE/variational-Lanczos implementation.

Selected backend-independent combinatorial helpers are also reused where appropriate.

This file remains the primary reference for checking whether the Qiskit variational-Lanczos implementation preserves the original algorithm.

---

## `n_site_graph_creation.py`

Creates the connectivity graphs used by the AIM ansatz.

---

# Fast environment check

From the repository root:

```bash
python -c "import numpy, scipy, qiskit, openfermion; import qiskit_port.dmft_qiskit as d; print('Core imports: OK')"
```

If Qiskit Aer is expected:

```bash
python -c "import qiskit_aer; print('Qiskit Aer: OK')"
```

Then syntax-check the main Qiskit files:

```bash
python -m py_compile \
    qiskit_port/qiskit_utils.py \
    qiskit_port/qiskit_ansatz.py \
    qiskit_port/qiskit_vqe.py \
    qiskit_port/dmft_qiskit.py
```

For the complete setup explanation, use `START_HERE.md`.

---

# Smoke test

Run:

```bash
python smoke_test_qiskit.py
```

The smoke test intentionally runs only the ground-state path.

It does **not** launch the expensive variational-Lanczos Green's-function calculation.

The approximate validated checkpoint is:

```text
system_size       = 4
seed              = 0
selected_depth    = 1
VQE energy        ~= -6.24852694
GS overlap error  ~= 2.1453e-02
overall_success   = True
```

The loose smoke-test threshold is deliberate.

Its purpose is to confirm that the environment, AIM initialization, exact solver, Qiskit VQE, state ordering, and comparison logic are working.

It is not a production-accuracy benchmark.

---

# Recommended production entry point

For the workflow that automatically increases ansatz depth until the ground-state target is reached:

```python
import qiskit_port.dmft_qiskit as dmft_qiskit

results = dmft_qiskit.run_gs_error_experiment_qiskit(
    system_size=4,
    seed=0,
    target_err=1e-4,
    gf_maxiter=int(1e6),
    gs_maxiter=int(1e6),
    gs_gtol=5e-4,
    gf_gtol=5e-5,
    pre_empt_layers=4,
    starting_depth=1,
    gs=False,
    display=True,
    plot=True,
    optimizer="BFGS",
    conv_tol=1e-6,
    optimizer_seed=0,
)
```

The high-level sequence is:

```text
L = starting_depth
        |
        v
optimize Qiskit ground state
        |
        v
compare with exact state
        |
        +---- target not met ----> increase L
        |
        v
target met
        |
        v
accept depth
        |
        v
construct phi- and phi+
        |
        v
variational Lanczos
        |
        v
G(omega)
        |
        v
A(omega)
```

where the spectral function is:

$$
A(\omega)
=
-\frac{1}{\pi}
\operatorname{Im}G(\omega).
$$

---

# Ground-state-only depth search

During development, it is often better to stop after the ground-state stage.

Use:

```python
results = dmft_qiskit.run_gs_error_experiment_qiskit(
    system_size=4,
    seed=0,
    target_err=1e-4,
    gs_maxiter=int(1e6),
    gs_gtol=5e-4,
    pre_empt_layers=4,
    starting_depth=1,
    gs=True,
    display=True,
    optimizer="BFGS",
    optimizer_seed=0,
)

print(results["selected_depth"])
print(results["depth_history"])
```

For the validated $N=4$, `seed=0`, `target_err=1e-4` case, the first accepted depth is expected to be:

```text
L = 4
```

See `DEVELOPER_NOTES.md` for the full depth table and regression values.

---

# Fixed-depth run

For controlled tests where the ansatz depth must be fixed explicitly:

```python
results = dmft_qiskit.run_aim_qiskit(
    system_size=4,
    seed=0,
    vqe_depth=4,
    optimizer="BFGS",
    gs_gtol=5e-4,
    gf_gtol=5e-5,
    conv_tol=1e-6,
    maxiter=int(1e6),
    calculate_exact=True,
    display=True,
    plot=True,
)
```

Use a fixed-depth run for:

- backend comparisons,
- profiling,
- reproducing a known depth,
- controlled validation,
- and debugging a specific ansatz layer count.

Use the automatic-depth function for normal production-style searching.

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

Do not apply an additional manual removal sign in notebook code.

---

## Complex Lanczos coefficients

The implementation may encounter a complex quantity before computing a Lanczos $b_i$ coefficient.

Only tiny numerical imaginary noise is removed.

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

This is based on the absolute overlap, not the squared fidelity.

---

# Known-good regression reference

For the primary validated system:

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
-6.34249498827641
```

and the automatic depth search should first accept:

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

Detailed numerical values are intentionally kept in `DEVELOPER_NOTES.md` so this README remains compact.

---

# Performance

The main Qiskit runtime cost is repeated statevector generation during optimization.

The final port includes several optimizations:

- parameterized ansatz-template caching,
- parameter binding instead of rebuilding circuits,
- Qiskit Aer statevector simulation when available,
- cached state-ordering permutations,
- precomputation of fixed Hamiltonian-vector products inside a Lanczos optimization.

Qiskit remains slower than Qulacs for this repeated native-statevector workload.

That is a performance limitation, not a known correctness failure.

Benchmark values and profiling history are in `DEVELOPER_NOTES.md`.

---

# Dependencies

The exact package versions are not frozen in this README.

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

Before upgrading major packages, especially NumPy, Qiskit, SciPy, or OpenFermion, run the documented regression checks afterward.

The project previously encountered compatibility failures after dependency changes, so environment changes should be treated as code changes and validated accordingly.

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
```

After a change, validate from cheapest to most expensive:

```text
1. imports / py_compile
2. smoke test
3. automatic GS depth search
4. controlled ground-state reconstruction
5. phi- / phi+ comparison
6. fixed-theta Lanczos cost
7. removal/addition coefficient chains
8. full controlled Green's function
9. independent production run
```

The detailed expected values are in `DEVELOPER_NOTES.md`.

---

# Handoff file map

```text
README.md
    Project front page and quick-reference navigation.

START_HERE.md
    Detailed onboarding, theory, operations, setup, and troubleshooting.

DEVELOPER_NOTES.md
    May--September debugging history + combined validation record.

VALIDATION.md
    Compatibility pointer to DEVELOPER_NOTES.md.

aim_tutorial_qiskit.ipynb
    Executable Qiskit tutorial matching the original Qulacs tutorial structure.

smoke_test_qiskit.py
    Fast ground-state-only environment/regression test.

qiskit_port/qiskit_utils.py
    Low-level numerical helpers.

qiskit_port/qiskit_ansatz.py
    Qiskit ansatz construction.

qiskit_port/qiskit_vqe.py
    Qiskit VQE and variational-Lanczos backend.

qiskit_port/dmft_qiskit.py
    High-level Qiskit workflow driver.
```

---

# Recommended reading order

```text
README.md
    |
    v
START_HERE.md
    |
    v
smoke_test_qiskit.py
    |
    v
aim_tutorial_qiskit.ipynb
```

When debugging or modifying internals:

```text
DEVELOPER_NOTES.md
```

The central maintenance rule is:

> **First prove that Qiskit and Qulacs agree at the same parameters. Then investigate differences between independent optimizer runs.**

That distinction prevented optimizer-path differences, ansatz limitations, state-ordering issues, and genuine implementation bugs from being confused with one another during the port.
