# AIM Qiskit Port — Developer Notes, Debugging History, and Validation Record

This is the combined developer-history and validation document for the Qiskit port of the Anderson Impurity Model (AIM) VQE + variational-Lanczos Green's-function workflow.

The purpose of this file is different from `START_HERE.md`:

- `START_HERE.md` explains how the code works and how to get started.
- **This file records how the code got to its current state, what broke along the way, how each problem was isolated, what tests were run, what fixes were made, and what numerical values should still be reproducible.**

This history covers the project from the initial codebase study in **May 2026** through the final Qiskit/Qulacs validation and handoff work in **September 2026**.

The record below is as complete as the preserved project conversations, files, notebooks, and validation outputs allow. Some early experiments were exploratory and used temporary settings; those are labeled as such so that they are not mistaken for final regression targets.

---

# 1. Final conclusion of the debugging project

The main scientific conclusion is:

> **When Qiskit and Qulacs are given the same variational state parameters, the two implementations reproduce the same AIM variational-Lanczos mathematics to floating-point precision.**

The final controlled tests established agreement in:

- the qubit Hamiltonian,
- the VQE ground-state reconstruction,
- statevector amplitudes after basis-order correction,
- particle-removal and particle-addition seed states,
- Lanczos expectation values,
- individual Lanczos cost evaluations,
- removal-branch coefficients,
- addition-branch coefficients,
- continued-fraction Green's functions,
- and spectral functions.

The large discrepancies that appeared during development were ultimately traced to several **different classes of problems**, not one single backend bug:

1. environment/package problems,
2. comparing different Green's-function algorithms,
3. invalid test setups using unsolved/random states or hard-coded sectors,
4. statevector basis-ordering mistakes,
5. removal/addition sign-convention mistakes,
6. different optimizer initial conditions,
7. optimizer path sensitivity to machine-precision numerical noise,
8. insufficient ansatz depth,
9. branch RNG coupling,
10. performance overhead in Qiskit statevector generation,
11. intermediate source-code structure/syntax mistakes,
12. notebook/API comparison issues.

The final implementation contains explicit protections or documented conventions for each of these.

---

# 2. Final source-code architecture

The Qiskit port was intentionally separated into layers:

```text
dmft_qiskit.py
      |
      v
qiskit_vqe.py
      |
      v
qiskit_utils.py
      |
      +--> qiskit_ansatz.py
```

The architecture rule established during the port was:

```text
qiskit_utils.py
    -> low-level numerical helpers

qiskit_ansatz.py
    -> circuit construction

qiskit_vqe.py
    -> VQE, Krylov, Lanczos, Green's-function backend

dmft_qiskit.py
    -> workflow/orchestration

aim_tutorial_qiskit.ipynb
    -> user-facing tutorial
```

Lower layers should not import higher layers.

The original project is still used for backend-independent functionality:

- `dmft.py` supplies system initialization and the original Qulacs reference workflow.
- `exact.py` supplies exact diagonalization and exact Green's-function calculations.
- selected purely combinatorial helpers from `vqe.py` remain reusable.
- `n_site_graph_creation.py` remains the graph/connectivity source.

This was a deliberate decision. The Qiskit port should replace backend-specific circuit/state operations, not duplicate working physics code without a reason.

---

# 3. May 2026 — understanding the original repository

## 3.1 Initial task

The first phase was not porting. It was understanding what the existing Qulacs repository actually did.

The original modules under study included:

```text
dmft.py
anderson_impurity_model.py
n_site_graph_creation.py
exact.py
vqe.py
```

The original workflow was mapped as:

```text
generate AIM parameters
    |
    v
construct AIM Hamiltonian
    |
    v
fermion -> qubit/Jordan-Wigner representation
    |
    +-------------------------+
    |                         |
    v                         v
exact ground state          VQE ground state
    |                         |
    +----------- compare ------+
                              |
                              v
                    particle removal/addition
                              |
                              v
                       Lanczos/Krylov
                              |
                              v
                     continued fraction
                              |
                              v
                         G(omega)
                              |
                              v
                      relative error / save
```

Important high-level functions identified in `dmft.py` included:

```text
initialize_system()
calculate_gs()
calculate_gf()
compare_gs()
calculate_rel_errors()
save_to_file()
run_gs_error_experiment()
```

The model parameters being traced included quantities such as:

```text
himp
uimp
vhyb
ebath
```

The important result of the May phase was that the original Green's-function workflow was understood to be **variational Lanczos**, not simply exact matrix inversion and not ordinary classical Lanczos.

That distinction became critical later.

---

## 3.2 Early code-reading concerns

While reading the repository, a few structural oddities were noted, including:

- duplicated/emulator-style methods in parts of the model/circuit code,
- an unusual `symmetric_ansatz_test(self, ...)` style calling pattern,
- substantial coupling between workflow logic and Qulacs-specific objects.

These observations were not themselves proof of numerical bugs. They were useful warnings that a line-by-line mechanical translation could easily preserve accidental structure while obscuring the underlying algorithm.

The port therefore evolved toward a layered Qiskit implementation rather than inserting Qiskit objects randomly throughout the original files.

---

# 4. Early June 2026 — environment, Jupyter, and HPC problems

Before meaningful numerical validation was possible, the environment had to be made reproducible.

This caused several real failures.

---

## 4.1 Local repository/Jupyter setup

The local repository was being run from a path such as:

```text
/Users/aporter2/Downloads/nlr_qc/qcaob-aim
```

The initial workflow depended on:

- the correct Python environment,
- the correct Jupyter kernel,
- the repository root being on the Python import path.

A recurring setup lesson was:

> A notebook can be open successfully while still using the wrong Python kernel.

If the notebook kernel and the environment containing the AIM packages do not match, imports can fail even though the shell environment looks correct.

The local setup sequence therefore became approximately:

```bash
python3 -m venv aim_env
source aim_env/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name aim_env
jupyter notebook
```

Later HPC work primarily used the conda environment `aimenv`.

---

## 4.2 Kestrel interactive/Jupyter workflow

The HPC workflow required:

```text
SSH to Kestrel
    |
    v
request interactive allocation
    |
    v
load Anaconda module
    |
    v
activate personal environment
    |
    v
start Jupyter on compute node
    |
    v
SSH port-forward from local machine
```

A working allocation in the preserved notes included:

```text
allocation: 14176415
node: x1004c7s0b1n1
```

The exact allocation/node is historical only; it should not be treated as a current command target.

The important lesson is that Jupyter should run on the allocated compute node, while the browser connects through SSH tunneling.

---

## 4.3 Shared-Anaconda `semver` PermissionError

### Symptom

Installing or modifying packages inside the shared Anaconda installation triggered a permissions failure involving `semver`, with packages under a shared `/nopt/.../site-packages` path.

### Cause

The shared module installation was not writable by the user.

The package manager was trying to modify a centrally managed Python environment.

### Fix

Create and use a **personal** environment instead of writing into the shared Anaconda environment.

The working environment became:

```text
aimenv
```

### Lesson

Do not try to "repair" shared HPC Python installations in place.

For future setup:

```bash
module load anaconda3
conda activate aimenv
```

or recreate a user-owned environment if needed.

---

## 4.4 NumPy 2.x `Inf` import failure

### Symptom

The code failed with an import error equivalent to:

```text
ImportError: cannot import name 'Inf' from numpy
```

### Cause

The existing code/dependencies expected an older NumPy API, while the environment had NumPy 2.x.

### Fix

The stable environment was restored by pinning:

```bash
numpy<2
```

with NumPy `1.26.4` being the expected compatible version in the debugging notes.

A source-level compatibility edit such as:

```python
from numpy import inf as Inf
```

was also identified as a possible narrow compatibility patch, but the safer project-level fix was to use the tested NumPy version rather than making scattered dependency patches.

### Lesson

Package-version changes can look like algorithm failures. Confirm the environment before debugging physics code.

---

# 5. June 2026 — establishing the original Qulacs baseline

Before porting to Qiskit, the original Qulacs workflow had to be run and understood numerically.

The requested baseline systems were:

```text
N = 2, 3, 4, 5
```

with both:

```text
exact/VQE ground states
exact/VQE Green's functions
```

where feasible.

The project convention uses energies and frequencies in eV, so the Green's function carries inverse-energy units, eV$^{-1}$.

---

## 5.1 Early system-size observations

One early set of runs reported approximate ground-state overlaps:

```text
N=2: 0.99994
N=3: 0.99751
N=4: 0.97855
N=5: approximately 0 under that early configuration
```

Early Green's-function relative errors included approximately:

```text
N=2: 1.343%
N=3: 2.912%
```

For the larger early cases, the Green's-function stage could be skipped when the ground-state accuracy gate was not met.

This taught an important workflow behavior:

> `run_gs_error_experiment()` does not guarantee a Green's-function calculation. It first requires the ground-state stage to satisfy its acceptance logic.

---

## 5.2 Detailed four-site baseline

For the important `N=4`, `seed=0` system:

```text
number of sites = 4
number of qubits = 8
exact charge = 4
exact spin = 0
```

The exact ground-state energy was:

```text
E_exact = -6.3424949882764 eV
```

An early depth scan gave approximately:

| Depth | VQE energy | Overlap | GS error |
|---:|---:|---:|---:|
| 1 | -6.2485269399 | 0.978547 | 2.1453e-2 |
| 2 | -6.3402543794 | 0.999789 | 2.108e-4 |
| 3 | -6.3407215658 | 0.999874 | 1.261e-4 |

The target being tested was:

```text
target_err = 1e-4
```

Depth 3 was close but did **not** satisfy that threshold.

This was initially easy to misread as a failure because the high-level call could finish without producing the expected Green's-function output.

It was not a crash.

It was the acceptance logic doing what it was written to do.

Later work confirmed that the next depth, `L=4`, is the first depth that meets the documented `1e-4` target for the final tested settings.

---

## 5.3 Scaling limitation

Early scaling experiments also demonstrated the expected exponential difficulty of exact/statevector calculations.

Representative runtime notes included:

```text
N=2: approximately 0.01 s for a small exact/reference step
N=4: approximately 7.6 s in an early benchmark
```

An attempted much larger case around `N>=11` failed during the early scaling work.

The precise failure mechanism was not preserved as a final validated diagnostic, so it should not be labeled as a specific software bug.

The correct lesson is:

> Exact diagonalization and full-statevector methods scale exponentially. A large-system failure is not automatically evidence of a Qiskit-port defect.

---

# 6. Mid-June 2026 — first Qiskit ground-state reproduction

The first Qiskit milestone was to reproduce the Hamiltonian and ground-state VQE before attempting the much more complicated Green's-function workflow.

---

## 6.1 Hamiltonian conversion test

A small test model was converted to Qiskit while preserving the original repository Hamiltonian.

A preserved exact-energy checkpoint was:

```text
E_exact = -5.065838993119 eV
```

The Qiskit Hamiltonian produced the same exact energy as the repository Hamiltonian.

This was the first strong confirmation that the fermion-to-qubit physics had not been altered by the Qiskit circuit port.

---

## 6.2 Symmetry-preserving ansatz reconstruction

For an early `N=2` test, the symmetry-preserving ansatz had:

```text
8 variational parameters
```

The Qiskit Statevector VQE reproduced the original result to roughly the fifth decimal place, with differences on the order of approximately $10^{-5}$ to $10^{-6}$ eV in the early independent optimization.

At this stage, that was sufficient to show:

```text
Hamiltonian representation: consistent
ansatz structure: consistent
VQE ground-state path: basically working
```

It was **not yet** sufficient to prove the full variational-Lanczos implementation.

---

# 7. Late June / July 2026 — the first Green's-function Qiskit implementations

A major conceptual problem appeared during the first Green's-function comparisons:

> The Qiskit code and the Qulacs code were not yet using the same Green's-function algorithm.

Three different paths existed.

---

## 7.1 Original Qulacs path

```text
VQE ground state
    |
    v
variational Lanczos
    |
    v
continued fraction
    |
    v
G(omega)
```

Each Krylov/Lanczos state is approximated variationally.

---

## 7.2 Qiskit v1: direct matrix inversion

The first Qiskit Green's-function benchmark used a direct resolvent:

$$
G(\omega)
=
\langle\phi|
(zI-H')^{-1}
|\phi\rangle,
$$

where:

$$
z=\omega+i\eta
$$

and:

$$
H'=H-E_0I.
$$

This is a useful small-system correctness benchmark, but it is not the same algorithm as the original variational-Lanczos code.

---

## 7.3 Qiskit v2: classical Lanczos

The next Qiskit implementation used ordinary classical Lanczos:

$$
\mathcal{K}
=
\operatorname{span}
\{
|\phi\rangle,
H|\phi\rangle,
H^2|\phi\rangle,
\ldots
\}.
$$

The resulting tridiagonal coefficients were used in the standard continued fraction.

For small systems and enough Krylov steps, this should closely reproduce direct matrix inversion.

That is exactly what was observed.

---

## 7.4 Why the Qiskit plots initially differed from Qulacs

At first, a difference between:

```text
Qiskit matrix-inverse/classical-Lanczos plots
```

and:

```text
original Qulacs variational-Lanczos plot
```

looked like a backend disagreement.

It was not.

The post-VQE algorithms were different.

The Qiskit versions were using exact/classical Krylov information after the VQE state, while the original repository variationally approximated every Krylov state.

This was an important turning point.

The project goal was redefined as:

> Port the **variational Lanczos** algorithm itself, not merely reproduce a Green's function using another classical method.

Functions identified as the essential porting targets included the equivalents of:

```text
lanczos_cost_function()
vqe_ideal_lanczos_iterations()
vqe_gf_pm()
vqe_gf()
```

---

# 8. July 2026 — architecture refactor

As the Qiskit implementation grew, putting every helper into one file became difficult to reason about.

The code was reorganized into the final layered structure:

```text
qiskit_utils.py
    -> qiskit_ansatz.py
    -> qiskit_vqe.py
    -> dmft_qiskit.py
    -> notebook
```

The design requirements were:

1. low-level helpers should have one implementation,
2. lower modules should not import higher workflow modules,
3. exact/backend-independent original code should be reused rather than duplicated,
4. notebook code should not contain hidden algorithmic sign/order fixes that belong in the backend.

This refactor later made it much easier to isolate errors because each test could target one layer.

---

# 9. July 2026 — an invalid 8-qubit comparison

One early eight-qubit comparison used:

- a random or unsolved ground-state parameter vector,
- hard-coded charge/spin sectors,
- and a Green's-function comparison built on top of that state.

The resulting Qulacs/Qiskit spectra were therefore **not diagnostically valid**.

The problem was not necessarily the code; the inputs did not represent the same solved physical state.

### Fix

The comparison procedure was changed to:

```text
solve or reconstruct a validated ground state
    |
    v
derive charge/spin sectors from that state/workflow
    |
    v
construct phi- and phi+
    |
    v
compare the two backends
```

### Lesson

Never use a visually plausible spectrum as evidence of backend correctness if the two backends were not given physically equivalent initial states and sectors.

---

# 10. July 2026 — independent optimizer starts created misleading coefficient differences

An early addition-branch test produced coefficients similar to:

```text
Qiskit a = [7.3302, 11.5073, 2.8913]
Qulacs  a = [7.3302, 10.9021, 1.6315]

Qiskit b = [0, 5.8927, 3.2297]
Qulacs  b = [0, 5.8927, 3.3066]
```

The first coefficients agreed well, but later coefficients diverged.

The initial suspicion was that the Qiskit Lanczos mathematics was wrong.

A fixed-parameter test showed something else:

> At identical circuit parameters, the Qiskit and Qulacs states and cost values agreed.

The later difference was being introduced by the optimization path.

This led to the general debugging rule used for the rest of the project:

```text
First compare the backends at identical theta.
Only after that compare independent optimizer trajectories.
```

Controlled tests are the only clean way to separate backend implementation errors from optimizer path differences.

---

# 11. July 2026 — Green's-function removal sign bug

This was a real calling-code bug.

In one comparison notebook:

```text
Qulacs removal:
    used w = -w
    then applied the removal sign before combining

Qiskit removal:
    used plain w
```

The two backends were therefore evaluating different mathematical functions.

That difference alone could shift or mirror spectral peaks.

---

## 11.1 Diagnostic convention

One diagnostic script expressed the Qulacs logic as:

```python
g_minus_raw = vqe_gf_pm(..., w=-w)
g_minus = -g_minus_raw
g_total = g_minus + g_plus
```

---

## 11.2 Final centralized production convention

The final Qiskit implementation centralizes the equivalent sign logic as:

```text
removal branch:
    continued fraction evaluated at -w

addition branch:
    continued fraction evaluated at +w

final combination:
    G_total = G_plus - G_minus_raw
```

or mathematically:

$$
G(\omega)
=
G_+(\omega)
-
G_-(\omega).
$$

The critical rule is:

> Do the removal sign correction **once**, in the Green's-function implementation. Do not repeat it in notebook/user code.

---

# 12. July/August 2026 — statevector ordering became the main correctness issue

A quantum state can be mathematically correct but appear wrong if its amplitudes are indexed in a different bit order.

This became one of the most important problems in the project.

---

## 12.1 The symptom

At different points, the Qiskit state could have:

```text
correct norm
plausible energy
wrong overlap with the exact/Qulacs state
```

That combination strongly indicated an indexing/ordering problem rather than a physically different state.

---

## 12.2 The two representations

The project needed to distinguish:

```text
raw Qiskit/Qulacs circuit ordering
```

from:

```text
OpenFermion/Python Hamiltonian-matrix ordering
```

The validated conversion is performed by:

```python
qulacs_to_python_ordering_qiskit(
    state_raw,
    n_qubits,
)
```

The conversion is effectively the required bit-index reversal/permutation.

---

## 12.3 The important storage rule

The variational Krylov list `ut` must remain in **raw circuit ordering**.

When matrix arithmetic is needed:

```python
state_ordered = qulacs_to_python_ordering_qiskit(
    state_raw,
    n_qubits,
)
```

should be a temporary converted copy.

Do not permanently store an ordered state back into `ut`.

---

## 12.4 A diagnostic that accidentally mixed representations

During one debugging test, an already reordered state was inserted into the raw Krylov chain.

Later Lanczos states then showed a large artificial mismatch.

This was not a physical failure.

The chain had mixed two incompatible basis conventions.

After restoring the raw-state invariant, the controlled backend comparison returned to machine-precision agreement.

---

# 13. August 2026 — independent-run Krylov fidelity looked poor

In independent optimizer runs, the removal branch showed results such as:

```text
phi norm:
    Qiskit approximately 0.17922
    Qulacs approximately 0.17920
```

but later independently optimized Krylov-state fidelities could degrade substantially, for example approximately:

```text
F(u1) ~ 0.972
F(u2) ~ 0.516
F(u3) ~ 0.713
```

This initially looked like the Qiskit Lanczos recursion was drifting away from Qulacs.

A same-angle test changed the interpretation:

```text
same-angle u1 fidelity ~ 1
```

Therefore:

```text
the circuit implementation could reproduce the same state
```

while:

```text
independent optimizers could land at different parameter vectors
```

This was a precursor to the more detailed September optimizer-divergence audit.

---

# 14. Diagnostic Hamiltonian shift tests

Some low-level branch tests deliberately used a simplified fixed shift such as:

```python
VQE_GS_ENERGY = -1.0
```

instead of rerunning the full ground-state optimization.

This was not intended as a physically final ground-state result.

It was a debugging technique used to hold the shifted Hamiltonian fixed while testing:

```text
Hamiltonian matrix conversion
phi construction
a0 / b1
cost function
optimizer trajectories
```

The rule for future developers is:

> Do not copy a diagnostic `VQE_GS_ENERGY=-1.0` harness into a production result and interpret the resulting spectrum as a physical benchmark.

---

# 15. September 2026 — systematic addition-branch audit

After the removal branch had been audited, the same step-by-step procedure was applied to the addition branch.

The goal was:

> Locate the **first exact operation** where Qiskit and Qulacs stopped agreeing.

The system used in the detailed addition audit had:

```text
n_qubits = 8
impurity_orbital = 2
```

---

# 16. Addition audit: seed-state test

The normalized particle-addition state was constructed in both backends.

Observed unnormalized norm:

```text
Qulacs ||phi+|| = 0.9838067763696449
Qiskit ||phi+|| = 0.9838067763696449
```

After normalization:

```text
fidelity = 1.0
max phase-aligned amplitude difference
    ~= 1.39e-16
L2 difference
    ~= 2.02e-16
```

### Conclusion

The addition-state construction itself was correct.

Any later disagreement had to occur after:

```text
u0 = phi+
```

was constructed.

---

# 17. Addition audit: first Lanczos coefficients before optimization

For:

$$
u_0=\phi_+,
$$

the first diagonal coefficient agreed:

```text
a0 Qulacs = -1.8103711954942607
a0 Qiskit = -1.8103711954942605
```

The Hamiltonian-square expectation also agreed:

```text
<H^2> Qulacs = 14.3576262554919
<H^2> Qiskit = 14.357626255492002
```

This produced:

```text
b1^2 Qulacs = 11.080182390016581
b1^2 Qiskit = 11.080182390016684

b1 Qulacs = 3.328690792190915
b1 Qiskit = 3.3286907921909306
```

### Conclusion

The addition branch was identical through:

```text
u0
a0
b1
```

before any variational optimization.

Therefore the first possible source of disagreement was the optimizer that constructs $u_1$.

---

# 18. Addition audit: charge/spin sector and ansatz configuration

The first addition-sector optimization used:

```text
n_up_plus = 3
n_down_plus = 2
charge_plus = 5
spin_plus = 1
```

The one-layer ansatz used:

```text
initial occupation indices = [0, 1, 3, 4, 7]
graph edges = 9
n_qubits = 8
n_params = 17
```

This verified that both backends were constructing the same optimization problem in the same symmetry sector.

---

# 19. Addition audit: same-parameter circuit/cost test

A fixed 17-parameter vector was supplied to both backends.

This removes the optimizer entirely.

Observed trial-state agreement:

```text
fidelity ~= 1.0
max phase-aligned amplitude difference ~= 1.2e-16
```

Observed cost:

```text
Qulacs = 20.35779760365056
Qiskit = 20.35779760365056
Delta cost = 0
```

### Conclusion

This ruled out:

```text
ansatz gate mismatch
parameter-order mismatch
statevector-generation mismatch at meaningful scale
Lanczos objective mismatch
transition-amplitude formula mismatch
overlap-term mismatch
```

at fixed $\theta$.

This was one of the strongest debugging results in the project.

---

# 20. Addition audit: same-start COBYLA test

COBYLA was used as a diagnostic optimizer because its evaluation path could be traced directly.

Settings included approximately:

```text
method = COBYLA
tol = 1e-6
maxiter = 2000
n_params = 17
```

Both backends began at:

```text
cost = 20.35779760365056
```

Both eventually ended with `MAXFUN`/status 3 in the traced test.

Representative best values were:

```text
Qulacs best fun = 0.05162897326858107
Qiskit best fun = 0.05128796389803612
```

The endpoints were close but not identical.

The next question was:

> At what exact evaluation did the optimizer paths diverge?

---

# 21. Addition audit: first microscopic divergence

The cost traces had:

```text
length Qulacs = 2000
length Qiskit = 2000
```

The first cost difference occurred at:

```text
evaluation 5
```

while the parameter vector was still bitwise identical.

The cost difference was only:

```text
2.66e-15
```

This is machine-precision scale.

The first parameter-vector difference appeared at:

```text
evaluation 18
```

The first large trajectory separation occurred at approximately:

```text
evaluation 182

max |Delta theta| ~= 3.60095e-3
|Delta cost|      ~= 8.94977e-4
```

Before that point, around evaluations 172--181:

```text
|Delta cost| <= about 1.68e-12
max |Delta theta| <= about 1.46e-12
```

### Conclusion

The backends did not begin with a meaningful mathematical disagreement.

Machine-precision differences slowly caused the derivative-free optimizer to choose different future points.

---

# 22. Where the first round-off difference came from

The evaluation-5 trial state was essentially the same.

The cost was recalculated three ways:

| Evaluation route | Cost |
|---|---:|
| Qulacs native operator arithmetic | 6.339806318365437 |
| Qulacs state + NumPy/Hmat | 6.339806318365440 |
| Qiskit state + NumPy/Hmat | 6.339806318365440 |

### Conclusion

One source of the first $10^{-15}$ discrepancy was the difference between native Qulacs operator arithmetic and the shared NumPy matrix route.

Using the same NumPy/Hamiltonian-matrix arithmetic removed that specific difference.

However, that did **not** guarantee identical long optimizer trajectories, because the Qulacs and Qiskit statevectors themselves can still differ at approximately $10^{-16}$ from floating-point implementation details.

Those differences are physically negligible but can affect a path-sensitive optimizer.

---

# 23. Cross-evaluation test: the decisive optimizer diagnostic

At a diverged evaluation point, the parameter vectors from the two backends were cross-evaluated:

```text
Qiskit cost at Qulacs theta
Qulacs cost at Qulacs theta

Qiskit cost at Qiskit theta
Qulacs cost at Qiskit theta
```

The two backends agreed to approximately floating-point precision at each fixed parameter vector.

### Final optimizer conclusion

> Independent optimizer trajectories can diverge even when the two backends implement the same objective function.

This is now a standard regression technique.

When a new discrepancy appears:

```text
DO NOT first compare final independently optimized theta.

FIRST cross-evaluate the cost at the same theta.
```

If the same-$\theta$ costs agree, the problem is not automatically a Qiskit physics bug.

---

# 24. The larger issue: one-layer ansatz expressivity

The optimizer-path difference was real, but it was not the dominant source of physical error in the first addition Krylov state.

The independently optimized one-layer states had approximately:

```text
Qulacs u1 fidelity to exact ~= 0.74016
Qiskit u1 fidelity to exact ~= 0.74056
```

Then the Lanczos objective was removed entirely and the ansatz was optimized directly for state fidelity.

Best one-layer direct fit:

```text
F ~= 0.77647
```

Six independent starts converged to essentially the same ceiling.

### Interpretation

The one-layer ansatz was not expressive enough to represent the exact first addition Krylov vector.

This was an ansatz limitation, not a backend limitation.

The one-layer case had:

```text
17 parameters
```

inside an addition-sector space with dimension approximately:

```text
24
```

---

# 25. Two layers resolved the first-Krylov-state expressivity problem

Increasing the ansatz depth from one layer to two gave:

```text
L=1 direct-fit fidelity = 0.77647
L=2 direct-fit fidelity = 0.999974
```

This corresponds to an approximately 99.99% reduction in relative infidelity for that test.

### Conclusion

The main first-step approximation error was strongly depth-dependent.

This result motivated:

```text
automatic depth search for the ground state
deeper controlled variational-Lanczos validation
```

and reinforced the rule:

> Do not blame the backend for an error that persists even when the optimizer is asked to fit the exact state directly.

---

# 26. Optimizer-option warnings

During earlier runs, SciPy emitted warnings similar to:

```text
OptimizeWarning: Unknown solver options: gtol, eps
```

when options intended for BFGS were passed to an optimizer that did not support them.

### Cause

Different SciPy optimizers accept different option dictionaries.

### Final rule

For the validated BFGS Green's-function path:

```python
{
    "maxiter": int(maxiter),
    "gtol": gf_gtol,
    "eps": 1e-7,
}
```

are appropriate.

`conv_tol` is passed as SciPy's `tol`.

Do not blindly pass BFGS-specific options to COBYLA or another optimizer.

COBYLA was valuable during debugging, but the production validation ultimately focused on the BFGS behavior used by the original workflow.

---

# 27. Complex Lanczos $b_i$ coefficients

Another point of concern was that the recurrence could produce a complex value for the quantity under the square root used to construct a Lanczos $b_i$.

A tempting "fix" would have been something like:

```python
np.sqrt(abs(b_squared))
```

or:

```python
np.sqrt(max(b_squared.real, 0))
```

That would silently change the algorithm.

Controlled tests showed that Qulacs and Qiskit could produce the same complex behavior.

### Final implementation

```python
b_squared = np.complex128(b_squared)

if abs(b_squared.imag) < 1e-12:
    b_squared = np.complex128(
        b_squared.real + 0.0j
    )

b[i] = np.sqrt(b_squared)
```

### Rule

Remove only demonstrably negligible floating-point imaginary noise.

Do **not** use absolute values or clipping unless intentionally changing the mathematics and then revalidating the whole chain.

---

# 28. Branch random-number coupling

The original variational Lanczos routine can retry an optimizer when `success=False`.

That means one branch may consume more random starting vectors than another.

If removal and addition use one shared RNG stream:

```text
extra removal retry
    ->
changes future random numbers
    ->
changes addition starting theta
```

This can create an apparent backend difference even though the addition code itself is unchanged.

### Fix

The Qiskit implementation uses independent branch seeds:

```text
seed_minus
seed_plus
```

The automatic driver can initialize both deterministically from `optimizer_seed`.

### Lesson

Random-start reproducibility must be managed per branch, not only globally.

---

# 29. Retry logic

Each variational Lanczos iteration can:

1. perform an initial optimization,
2. retry with a new random start if `success=False`,
3. make up to several additional attempts,
4. propagate the successful result when one succeeds,
5. otherwise retain the lowest-cost failed result.

Large function-evaluation counts are therefore not automatically a Qiskit-specific bug.

The original Qulacs path can also spend a large number of evaluations and retries on difficult Krylov states.

Do not remove retries merely to make a benchmark look faster unless the algorithm is intentionally being changed.

---

# 30. Performance investigation

Once correctness was established at fixed parameters, runtime became the next issue.

The question was:

> Is Qiskit slower because it is doing more optimizer work, or because each state/cost evaluation is more expensive?

The controlled benchmarks answered this.

---

# 31. Per-cost benchmark

For one controlled `N=4`, `L=2` Lanczos cost evaluation:

```text
Original Qiskit full cost ~= 2.871657 ms
Qulacs full cost          ~= 0.172157 ms
```

Approximate gap:

```text
16.7x
```

---

# 32. State-generation benchmark

The state-generation/state-update portion was approximately:

```text
Qiskit Statevector.from_instruction ~= 2.215008 ms
Qulacs update_quantum_state         ~= 0.041315 ms
```

Approximate gap:

```text
53.6x
```

These numbers do not conflict.

The approximately 54x number applies only to the state-generation operation.

The approximately 16.7x number applies to the full cost evaluation, which also contains NumPy arithmetic that is similar for both backends.

---

# 33. Same-start BFGS workload test

A controlled BFGS test showed:

```text
Qiskit:
    nit  = 57
    nfev = 2135

Qulacs:
    nit  = 57
    nfev = 2135
```

The final objective values agreed to about $10^{-12}$ or better, and cross-evaluation agreed to floating-point precision.

### Conclusion

The optimizer was not inherently taking more steps in Qiskit.

The dominant runtime penalty came from the cost of each Qiskit statevector/cost evaluation.

---

# 34. Performance fixes added to the final code

## 34.1 Parameterized ansatz template cache

Instead of rebuilding a complete numeric `QuantumCircuit` for every proposed $\theta$, the code caches one parameterized template per ansatz configuration.

The cache key includes:

```text
number of qubits
number of layers
initial occupations
connected-graph edge signature
compilation mode
```

Then each optimizer evaluation binds a new parameter vector into the existing template.

---

## 34.2 Qiskit Aer statevector backend

When available, the code uses:

```python
AerSimulator(method="statevector")
```

instead of relying only on:

```python
Statevector.from_instruction(...)
```

The implementation falls back safely when Aer is not installed.

---

## 34.3 Precompute fixed $H|u_{i-1}\rangle$

During optimization of one Krylov state, the previous Krylov vector is fixed.

Therefore:

$$
H|u_{i-1}\rangle
$$

does not need to be recalculated for every trial $\theta$.

The final path computes it once and passes it into the repeated cost function.

---

## 34.4 Cache the ordering permutation

The bit-reversal/permutation indices depend only on `n_qubits`.

They are therefore cached using an `lru_cache`-style mechanism rather than regenerated for every state.

---

## 34.5 Measured improvement

The cached parameterized-circuit + Aer path reduced the Qiskit full-cost time to roughly:

```text
1.15--1.23 ms
```

in the controlled benchmark.

That was roughly a:

```text
2.4x speedup
```

over the original Qiskit implementation.

It still did not make Qiskit as fast as Qulacs for repeated native statevector evolution.

That remaining difference is a performance limitation, not a known correctness failure.

---

# 35. Intermediate source-structure errors

During final refactoring, two code-structure problems appeared in intermediate files:

1. an indentation problem inside `run_aim_qiskit()`,
2. `_run_sector()` accidentally slipping outside the intended scope of `solve_vqe_qiskit()`.

These were ordinary source-merging/refactoring mistakes, not algorithmic discoveries.

### Fix

The final files were syntax-checked with:

```bash
python -m py_compile \
    qiskit_port/qiskit_utils.py \
    qiskit_port/qiskit_vqe.py \
    qiskit_port/dmft_qiskit.py
```

### Rule

After manual copying or refactoring, run syntax checks **before** launching a numerical optimization.

---

# 36. Controlled final ground-state validation

In the strongest controlled ground-state test, Qulacs optimized the `N=4`, `L=4` ansatz and Qiskit reconstructed the **same** optimized parameter vector.

Observed:

```text
Exact energy  = -6.34249498827641
Qulacs energy = -6.342310802334666
Qiskit energy = -6.3423108023346755
```

Backend energy difference:

```text
|E_Qiskit - E_Qulacs|
    = 9.769962616701378e-15
```

Qiskit/Qulacs state fidelity:

```text
0.9999999999999829
```

Maximum amplitude difference:

```text
1.1116099371267112e-15
```

Fidelity to exact:

```text
Qulacs-exact = 0.9999798399015069
Qiskit-exact = 0.9999798399015088
```

### Conclusion

The Qiskit ansatz/state reconstruction is backend-equivalent to Qulacs at floating-point precision.

---

# 37. Controlled final $\phi^-$ / $\phi^+$ validation

Using the same controlled `N=4`, `L=4` VQE ground state:

```text
phi- norm Qulacs = 0.2850349395137528
phi- norm Qiskit = 0.2850349395137527

phi+ norm Qulacs = 0.9585171272629309
phi+ norm Qiskit = 0.9585171272629319
```

Raw-state fidelities:

```text
phi- fidelity = 1.0
phi+ fidelity = 1.0
```

### Conclusion

The Jordan-Wigner particle-removal/addition seed-state logic is equivalent between backends.

---

# 38. Controlled final removal Lanczos chain

Qulacs optimized each Krylov state.

Qiskit then reconstructed the same selected $\theta$ values and recomputed the chain.

Observed full-chain differences:

```text
max |Delta a|  = 1.2442425513123377e-14
max |Delta b|  = 6.306066779870889e-14

mean |Delta a| = 4.943739580806205e-15
mean |Delta b| = 3.8244962752287395e-14
```

### Regression expectation

If the same states/parameters are supplied, removal coefficients should agree at floating-point scale.

---

# 39. Controlled final addition Lanczos chain

Using the same controlled procedure:

```text
max |Delta a|  = 1.2442425513123377e-14
max |Delta b|  = 5.3734794391857577e-14

mean |Delta a| = 5.532031580552665e-15
mean |Delta b| = 1.9495516312417748e-14
```

### Conclusion

The addition branch is also backend-equivalent at floating-point scale.

---

# 40. Hybrid/full Green's-function milestone

A later `N=4`, `L=2` hybrid full-Green's-function comparison reached approximately:

```text
max |Delta G| ~= 1.38e-12
```

between the Qiskit reconstruction and Qulacs reference path.

This was an important bridge between branch-level coefficient validation and the final `L=4` controlled spectrum.

---

# 41. Controlled final `N=4`, `L=4` Green's-function validation

Using the same optimized variational states in both backends:

```text
Qiskit GF relative error vs exact ~= 0.0945669
Qulacs GF relative error vs exact ~= 0.0945669
```

Difference between the two relative errors:

```text
~= 3.9e-14
```

Maximum spectral difference:

```text
max |A_Qiskit - A_Qulacs|
    ~= 7.0e-14
```

### Main variational peak

```text
omega ~= 5.33033
A(omega) ~= 1.4122641
```

### Main exact peak on the sampled grid

```text
omega ~= 5.28028
A_exact(omega) ~= 1.49584
```

### Final interpretation

The Qiskit and Qulacs variational results are the same.

The remaining approximately 9.46% Green's-function error relative to exact is a property of the variational approximation/ansatz for that run, not a Qiskit-vs-Qulacs implementation mismatch.

---

# 42. Why deeper ansatz depth mattered

An earlier controlled `N=4`, `L=2` Green's-function calculation had approximately:

```text
GF relative error ~= 0.137
```

At `L=4`:

```text
GF relative error ~= 0.0946
```

This is an approximately 31% reduction in the Green's-function relative error.

This is consistent with the earlier direct-fit finding that ansatz depth can be the dominant approximation limitation.

---

# 43. Automatic ansatz-depth search

The original Qulacs high-level workflow increases depth until the requested ground-state overlap-error target is satisfied.

The Qiskit port now reproduces this behavior with:

```python
run_gs_error_experiment_qiskit(...)
```

The acceptance quantity is:

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
1-
\mathrm{gs\_overlap}.
$$

Do not confuse this amplitude-overlap error with squared fidelity.

---

# 44. Final automatic-depth regression target

For:

```text
system_size = 4
seed = 0
target_err = 1e-4
optimizer = BFGS
gs_gtol = 5e-4
starting_depth = 1
pre_empt_layers = 4
```

the final Qiskit results were:

| Depth | VQE ground-state energy | GS overlap error | Optimizer success | Target met? |
|---:|---:|---:|:---:|:---:|
| 1 | -6.248526936637 | 2.1452866e-02 | True | No |
| 2 | -6.340252287092 | 2.1033579e-04 | True | No |
| 3 | -6.340721627947 | 1.2593785e-04 | True | No |
| 4 | -6.342237407455 | 1.7446567e-05 | True | Yes |

Expected final selection:

```text
selected_depth = 4
overall_success = True
```

Exact reference:

```text
E_exact = -6.34249498827641
```

### Important interpretation

`pre_empt_layers=4` means:

> stop searching after layer 4.

It does **not** mean:

> layer 4 is hard-coded as the correct physical answer.

For another system, seed, ansatz, or target, the accepted depth may differ.

---

# 45. Ground-state smoke-test target

The final handoff includes a fast ground-state-only smoke test.

It intentionally uses a loose acceptance target so it can stop at `L=1`.

Representative expectation:

```text
system_size       = 4
seed              = 0
selected_depth    = 1
VQE energy        ~= -6.24852694
GS overlap error  ~= 2.1453e-02
overall_success   = True
```

This is a **software/environment regression test**, not a claim that `L=1` is production-accurate.

The purpose is to verify:

```text
imports
system initialization
exact solver
Qiskit VQE
state ordering
comparison logic
result dictionary
```

without paying for a full Green's-function run.

---

# 46. Qulacs and Qiskit high-level API mismatch discovered during handoff

The original Qulacs function:

```python
dmft.run_gs_error_experiment(...)
```

returns:

```text
None
```

even though it calculates and plots `g_vqe` internally.

The Qiskit high-level workflow returns a results dictionary.

This created confusion when writing side-by-side comparison notebook cells.

A line such as:

```python
results_qulacs = dmft.run_gs_error_experiment(...)
```

does **not** produce a Qulacs results dictionary.

### Practical consequence

For Qulacs data comparisons, use one of these approaches:

1. call lower-level functions that expose the arrays,
2. modify the Qulacs function to return data,
3. or intercept/capture the arrays passed into `plot_gfs()`.

Do not assume the two high-level APIs return the same object shape.

---

# 47. Notebook variable-collision issue

The original tutorial frequently uses generic names such as:

```python
g_vqe
```

When both Qiskit and Qulacs are run in the same notebook, one backend can overwrite the other's array.

This led to `NameError` and mislabeling problems when comparison cells tried to use variables such as:

```text
g_qiskit
g_qulacs
```

without first explicitly creating them.

### Safe pattern

Immediately capture backend-specific results:

```python
g_qiskit = np.asarray(
    qiskit_result
).copy()

g_qulacs = np.asarray(
    qulacs_result
).copy()
```

or capture Qulacs plot inputs programmatically.

The lesson is broader:

> Never depend on a generic notebook variable after switching backends.

---

# 48. Side-by-side plot reproduction issue

A comparison plot initially did not look like the known Qulacs reference figure.

The causes included:

- plotting only the variational spectral curve rather than the original combination of curves,
- using inconsistent depth/system settings,
- mixing data from different runs,
- and not preserving the Qulacs figure's exact context.

The corrected comparison used the same:

```text
system size
seed
ansatz depth
frequency grid
broadening
```

and plotted both:

```text
VQE real
VQE spectral
exact real
exact spectral
```

for each backend.

The Qulacs reference figure being reproduced used a fixed-depth case, so the comparison had to match that depth rather than silently use a different automatic-depth result.

---

# 49. Peak-comparison issue

The first peak-comparison script simply selected the top few local maxima by height.

That can choose visually unimportant numerical bumps while missing the peaks a human would naturally call "the noticeable peaks."

### Fix

Use a prominence criterion, for example:

```python
from scipy.signal import find_peaks

indices, props = find_peaks(
    spectral,
    prominence=0.05 * np.max(spectral),
)
```

Then visually confirm the detected peaks before computing pairwise backend differences.

### Lesson

"Top N local maxima" and "physically/visually noticeable peaks" are not the same selection rule.

Peak-selection logic should be documented whenever peak positions/heights are used as validation evidence.

---

# 50. Tutorial reconstruction issue

The first generated Qiskit tutorial was much shorter and more redesigned than the original Qulacs tutorial.

It had approximately:

```text
37 cells
```

while the original tutorial had:

```text
66 cells
```

This did not satisfy the handoff goal.

The requirement was not merely:

> create a Qiskit tutorial.

It was:

> create a Qiskit tutorial that mirrors the original Qulacs tutorial structure so the next developer can compare the two directly.

### Fix

The original notebook structure was inspected cell by cell.

The Qiskit tutorial was rebuilt to preserve the same progression:

```text
imports
dmft ground state
input arguments
initialization
exact GS
VQE GS
Hamiltonian
reconstructed GS
energy/overlap comparison
high-level dmft call
checkpoint discussion
Green's-function dimensions
phi states
exact GF
VQE GF
GF errors
high-level GF call
bash/help section
```

Only backend-specific operations were replaced.

### Lesson

For a migration handoff, structural similarity is a feature. A cleaner redesign can be less useful than a faithful port.

---

# 51. Checkpointing/persistence difference

The original Qulacs workflow includes result/checkpoint file behavior.

The Qiskit port did not fully reproduce every persistence path during the first validated implementation.

The handoff intentionally prioritizes:

```text
correct end-to-end calculation
reproducible numerical validation
clear returned result dictionaries
```

over designing a new storage format prematurely.

Any future result-persistence implementation should be added only after preserving the regression tests in this file.

---

# 52. Controlled validation vs independent production runs

Two types of results appear throughout this document.

They must not be mixed.

---

## 52.1 Controlled backend validation

Procedure:

```text
Qulacs optimizes theta
    |
    v
Qiskit reconstructs the SAME theta
    |
    v
compare states/costs/coefficients/GF
```

Purpose:

> Test whether the Qiskit backend implements the same mathematics.

This is where machine-precision agreement was demonstrated.

---

## 52.2 Independent production run

Procedure:

```text
Qiskit performs its own random initialization
    |
    v
Qiskit optimizer follows its own trajectory
    |
    v
Qiskit accepts depth
    |
    v
Qiskit runs its own Lanczos optimizations
```

Purpose:

> Test whether the Qiskit workflow works independently end to end.

Two independent production runs can differ more than a controlled backend test because optimization is path-dependent.

A larger independent-run difference is not automatically an implementation regression.

---

# 53. Final invariant: state ordering

This is one of the most dangerous places to make a silent mistake.

Keep:

```text
raw circuit ordering
```

inside:

```text
Krylov states / ansatz-state overlap calculations
```

Convert to:

```text
OpenFermion/Python ordering
```

only when interacting with:

```text
Hamiltonian matrices
exact states
matrix expectation values
```

Do not insert converted states into the raw `ut` chain.

---

# 54. Final invariant: Green's-function signs

Use:

```text
removal -> -w
addition -> +w
G_total = G_plus - G_minus
```

Do not perform another manual removal sign correction outside the centralized function.

---

# 55. Final invariant: complex Lanczos coefficients

Do not replace:

```text
complex square root with tiny-imaginary cleanup
```

with:

```text
absolute value
clipping
real-only forcing
```

unless intentionally changing the algorithm and rerunning every controlled test.

---

# 56. Final invariant: optimizer comparisons

When Qiskit and Qulacs independently optimize to different states:

1. save both $\theta$ vectors,
2. evaluate each backend at the Qulacs $\theta$,
3. evaluate each backend at the Qiskit $\theta$,
4. compare states and costs at fixed parameters.

Only after this should the discrepancy be labeled an implementation problem.

---

# 57. Final invariant: branch RNG independence

Keep removal and addition random-start streams independent.

Otherwise a retry in one branch can change the other branch's starting conditions.

---

# 58. Final invariant: parameter ordering

The ansatz parameter vector must be consumed in the same gate/edge iteration order in both backends.

For the validated `N=4` one-layer graph:

```text
n_qubits = 8
n_edges = 9
parameters per layer = 17
```

Therefore:

```text
L=1 -> 17
L=2 -> 34
L=3 -> 51
L=4 -> 68
```

A change to graph edge iteration order can silently change the meaning of an existing $\theta$ vector.

---

# 59. Final invariant: exact comparison metric

The original ground-state metric is:

$$
\mathrm{gs\_overlap}
=
\left|
\langle
\psi_{\mathrm{VQE}}
|
\psi_{\mathrm{exact}}
\rangle
\right|,
$$

not the squared fidelity.

Then:

$$
\mathrm{gs\_error}
=
1-\mathrm{gs\_overlap}.
$$

If a report uses squared fidelity instead, label it explicitly.

---

# 60. Regression sequence after future changes

Run tests from cheapest to most expensive.

## Stage 1 — syntax/import

```bash
python -m py_compile \
    qiskit_port/qiskit_utils.py \
    qiskit_port/qiskit_vqe.py \
    qiskit_port/dmft_qiskit.py
```

Then verify imports.

---

## Stage 2 — smoke test

```bash
python smoke_test_qiskit.py
```

Expected approximate result:

```text
selected_depth = 1
overall_success = True
VQE energy ~= -6.24852694
GS error ~= 2.1453e-02
```

---

## Stage 3 — automatic ground-state depth search

Use:

```text
N=4
seed=0
target_err=1e-4
BFGS
starting_depth=1
pre_empt_layers=4
```

Expected:

```text
selected_depth = 4
```

---

## Stage 4 — controlled ground-state reconstruction

Expected backend fidelity:

```text
approximately 1
```

and energy difference around floating-point scale.

---

## Stage 5 — controlled $\phi^-$ / $\phi^+$

Expected:

```text
fidelity = 1
norms equal to numerical precision
```

---

## Stage 6 — one fixed-$\theta$ Lanczos cost

Expected:

```text
same trial state
same cost
```

to numerical precision.

---

## Stage 7 — full removal/addition coefficient chains

Expected `a_i` and `b_i` differences:

```text
approximately 1e-14 to 1e-13 scale
```

under controlled same-state reconstruction.

---

## Stage 8 — full controlled Green's function

Expected Qiskit/Qulacs spectral difference:

```text
approximately floating-point scale
```

with the known controlled `L=4` reference values above.

---

## Stage 9 — independent production workflow

Only after the controlled tests pass should independent optimizer differences be investigated.

---

# 61. Current known limitations

## 61.1 Qiskit runtime

Qiskit remains slower than Qulacs for repeated native statevector evolution in this workload.

The current implementation reduces avoidable overhead but does not remove that backend-level difference.

---

## 61.2 Exact scaling

Exact diagonalization and full statevector representations scale exponentially.

Large-system failure should be evaluated in the context of memory/state-space growth before being treated as a port regression.

---

## 61.3 Optimizer path sensitivity

Independent optimizers can diverge from machine-precision perturbations.

This is inherent to numerical optimization and remains relevant even after backend correctness is established.

---

## 61.4 Ansatz depth

Insufficient ansatz depth can dominate the physical error.

A one-layer optimizer cannot be expected to reproduce a state that lies outside the representational capacity of that ansatz.

---

## 61.5 Persistence/checkpoint parity

The scientific computation is the main validated target. File/checkpoint persistence is not as fully standardized in the Qiskit port as the numerical workflow.

---

# 62. Recommended future work

The safest next development work is:

1. preserve the current regression targets,
2. archive independent production `N=4` results with seeds/settings,
3. add a canonical environment file or version lock,
4. standardize result persistence,
5. add automated unit tests for:
   - state ordering,
   - phi construction,
   - fixed-$\theta$ cost equality,
   - sign convention,
   - continued fraction,
6. add an automated controlled Qiskit/Qulacs regression harness,
7. profile larger systems only after exact-reference scaling is separated from Qiskit runtime,
8. experiment with deeper ansatzes only while preserving the fixed-parameter backend tests.

---

# 63. "Do not reintroduce these bugs" checklist

Before merging a change, verify that none of the following happened:

```text
[ ] raw and OpenFermion state orderings were mixed
[ ] an ordered state was inserted into raw Krylov list ut
[ ] removal used +w instead of -w
[ ] removal sign was applied twice
[ ] Qiskit and Qulacs were compared using different theta
[ ] hard-coded sectors replaced derived sectors
[ ] a random/unsolved GS was used as a physical comparison state
[ ] one global RNG stream coupled removal and addition retries
[ ] abs()/clipping was inserted into b_squared
[ ] BFGS-only options were passed to another optimizer
[ ] generic g_vqe notebook variables were overwritten between backends
[ ] Qulacs run_gs_error_experiment() was assumed to return a dict
[ ] a fixed-depth plot was compared with a different-depth reference
[ ] top-N numerical local maxima were called "the noticeable peaks" without prominence/inspection
[ ] a source refactor was run without py_compile
[ ] a classical-Lanczos or matrix-inverse result was described as variational Lanczos
```

---

# 64. Key numerical regression table

| Check | Expected result |
|---|---|
| `N=4`, exact GS energy | `-6.34249498827641` |
| Automatic depth target `1e-4` | first accepted depth `L=4` |
| Controlled L4 backend energy difference | `~9.77e-15` |
| Controlled L4 Qiskit/Qulacs fidelity | `0.9999999999999829` |
| Controlled $\phi^-$ fidelity | `1.0` |
| Controlled $\phi^+$ fidelity | `1.0` |
| Removal max `|Delta a|` | `~1.24e-14` |
| Removal max `|Delta b|` | `~6.31e-14` |
| Addition max `|Delta a|` | `~1.24e-14` |
| Addition max `|Delta b|` | `~5.37e-14` |
| Controlled L4 GF error vs exact | `~0.0945669` |
| Max Qiskit/Qulacs spectral difference | `~7e-14` |
| Controlled variational main peak | `omega ~ 5.33033`, `A ~ 1.4122641` |
| Exact main peak | `omega ~ 5.28028`, `A ~ 1.49584` |
| Original Qiskit full cost | `~2.872 ms` |
| Qulacs full cost | `~0.172 ms` |
| Cached/Aer Qiskit full cost | `~1.15--1.23 ms` |
| Same-start BFGS workload | `nit=57`, `nfev=2135` for both |

---

# 65. Final handoff interpretation

The most important thing for the next developer to understand is that this project did **not** end with:

> "Qiskit produced a plot that looked approximately like Qulacs."

It ended with a much stronger result:

```text
same physical model
same sectors
same ansatz parameters
same phi states
same Lanczos objective
same a/b coefficients
same continued fraction
same Green's function
same spectrum
```

to numerical precision.

The remaining differences in independent runs are therefore interpreted through:

```text
optimizer trajectory
ansatz expressivity
requested depth/accuracy
runtime
```

before assuming the backend mathematics is wrong.

That distinction was the central lesson of the May--September debugging effort.
