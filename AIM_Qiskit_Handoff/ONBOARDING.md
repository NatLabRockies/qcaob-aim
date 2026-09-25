# 10-Minute Onboarding — Detailed Walkthrough

This document is the **first file a new developer should read** when taking over the Qiskit Anderson Impurity Model (AIM) port.

The commands themselves can still be completed in roughly ten minutes if the Python environment is already installed. The explanations are intentionally much more detailed than a ten-minute checklist so that this file can also serve as a reference later.

The purpose of the first ten minutes is **not** to run the entire Green's-function calculation. The purpose is to answer a simpler question:

> **Is the repository, environment, Qiskit backend, exact solver, VQE ground-state path, and state-comparison logic all working correctly before I spend significant time on a full variational-Lanczos calculation?**

If the answer is yes, then move on to `aim_tutorial_qiskit.ipynb` and eventually to the full production workflow.

---

# 1. First understand what this project is doing

The project solves an **Anderson Impurity Model (AIM)**. Conceptually, an AIM contains one interacting impurity site coupled to a set of bath sites. The model is represented by a Hamiltonian $H$, which is the operator that contains the energies and couplings that define the physical system.

For the convention used in this repository:

```text
system_size = N
```

means the AIM contains:

```text
1 impurity site + (N - 1) bath sites
```

Each spatial site has a spin-up and a spin-down degree of freedom. After the fermionic problem is mapped to qubits, the tested models therefore use:

```text
number of qubits = 2 * system_size
```

For example:

```text
system_size = 4
4 spatial sites
8 spin orbitals
8 qubits
```

The project calculates two major things:

```text
1. The ground state
2. The Green's function / spectral function
```

The **ground state** is the lowest-energy quantum state of the Hamiltonian.

The **Green's function** describes how the system responds when a particle is added or removed. From the retarded Green's function $G(\omega)$, the spectral function is

$$
A(\omega)
=
-\frac{1}{\pi}\operatorname{Im}G(\omega).
$$

The peaks in $A(\omega)$ correspond to important excitation energies and spectral weights.

This repository contains two ways of approaching the problem:

```text
Exact calculation
    vs.
Variational quantum calculation
```

The exact method is used as the classical reference for small systems. The variational method uses a parameterized quantum circuit and optimization.

The original repository performs the variational work with **Qulacs**. This handoff contains the **Qiskit port** of the same workflow.

---

# 2. The high-level Qiskit workflow

The production workflow can be summarized as:

```text
Create AIM
    |
    v
Construct qubit Hamiltonian
    |
    +----------------------+
    |                      |
    v                      v
Exact ground state      Qiskit VQE ground state
    |                      |
    +---------- compare ---+
                           |
                           v
                 Accept an ansatz depth
                           |
                           v
                 Construct phi- and phi+
                           |
              +------------+------------+
              |                         |
              v                         v
       Removal branch             Addition branch
       variational Lanczos        variational Lanczos
              |                         |
              +------------+------------+
                           |
                           v
                      G_Qiskit(omega)
                           |
                           v
                       A_Qiskit(omega)
                           |
                           v
               Compare with exact result
```

The first ten-minute smoke test intentionally stops after the **ground-state comparison**.

Why? The Green's-function calculation contains multiple variational Lanczos optimizations. Each optimization can require many evaluations of a parameterized quantum circuit. If the environment or state ordering is wrong, discovering that only after a long Green's-function run wastes time.

---

# 3. Repository map: what each important file does

The Qiskit port is separated into layers.

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

The main files mean the following.

| File | Purpose |
|---|---|
| `qiskit_port/dmft_qiskit.py` | High-level workflow. This is normally the file a notebook or user calls. |
| `qiskit_port/qiskit_vqe.py` | Qiskit VQE, statevector simulation, Krylov-state construction, variational Lanczos, and Green's-function branches. |
| `qiskit_port/qiskit_utils.py` | Low-level numerical helpers: expectation values, overlaps, transition amplitudes, continued fractions, state-order conversion, Hamiltonian matrices, and initial occupations. |
| `qiskit_port/qiskit_ansatz.py` | Builds the parameterized symmetry-preserving quantum circuit. |
| `dmft.py` | Original high-level Qulacs workflow. Also contains backend-independent system initialization reused by the Qiskit port. |
| `vqe.py` | Original Qulacs VQE/Lanczos code. The Qiskit workflow still reuses selected backend-independent combinatorial helpers from this file. |
| `exact.py` | Exact diagonalization and exact Green's-function routines. These are backend-independent and are reused directly. |
| `n_site_graph_creation.py` | Builds the graph structure describing which qubits/sites are connected in the ansatz. |
| `aim_tutorial_qiskit.ipynb` | Qiskit counterpart to the original Qulacs tutorial. |
| `VALIDATION.md` | Numerical reference values and backend-equivalence checks. |
| `DEVELOPER_NOTES.md` | Important implementation decisions, debugging conclusions, and conventions that should not be changed casually. |
| `smoke_test_qiskit.py` | Fast ground-state-only environment and regression check. |

The normal rule is:

```text
User/notebook code
    calls dmft_qiskit.py

dmft_qiskit.py
    coordinates the calculation

qiskit_vqe.py
    performs Qiskit variational work

qiskit_utils.py
    performs low-level numerical operations
```

---

# 4. Minute 0–2: activate the Python environment

## 4.1 Activate the environment

Use the Python environment that already contains the AIM dependencies.

Example:

```bash
conda activate aimenv
```

### What this command means

`conda activate aimenv` tells the shell:

> Use the Python interpreter and installed packages contained inside the environment named `aimenv`.

This matters because the repository depends on packages such as:

```text
NumPy
SciPy
Qiskit
OpenFermion
NetworkX
Matplotlib
Qiskit Aer (recommended)
```

If the wrong environment is active, Python may either fail to import a package or may import an incompatible version.

You can check which Python executable is active with:

```bash
which python
```

On Windows PowerShell, use:

```powershell
Get-Command python
```

---

## 4.2 Move to the repository root

Example:

```bash
cd /path/to/qcaob-aim
```

### What "repository root" means

The repository root is the directory containing the original project files, for example:

```text
qcaob-aim/
├── dmft.py
├── exact.py
├── vqe.py
├── n_site_graph_creation.py
├── qiskit_port/
│   ├── dmft_qiskit.py
│   ├── qiskit_vqe.py
│   ├── qiskit_utils.py
│   └── qiskit_ansatz.py
└── ...
```

Running from the repository root ensures Python can find both the original modules and the `qiskit_port` package.

---

# 5. Minute 0–2: check the important imports

Run:

```bash
python -c "import numpy, scipy, qiskit, openfermion; import qiskit_port.dmft_qiskit as d; print('Core imports: OK')"
```

If everything imports, the final line should be:

```text
Core imports: OK
```

## What each import is used for

### `numpy`

NumPy handles numerical arrays and linear algebra.

Examples in this project include:

$$
\langle \psi | H | \psi \rangle
$$

matrix-vector products,

```python
Hmat @ state
```

vector norms,

```python
np.linalg.norm(state)
```

and complex arrays representing quantum states and Green's functions.

---

### `scipy`

SciPy provides the classical optimizer used by VQE and variational Lanczos.

The main optimizer used in the validated workflow is:

```text
BFGS
```

BFGS repeatedly changes the circuit parameters $\theta$ to reduce a cost function.

For the ground state, the cost is essentially the energy:

$$
E(\theta)
=
\langle \psi(\theta)|H|\psi(\theta)\rangle.
$$

The optimizer is trying to find

$$
\theta_{\min}
=
\arg\min_\theta E(\theta).
$$

---

### `qiskit`

Qiskit represents and manipulates the quantum circuits used by the port.

The ansatz circuit is a parameterized circuit:

$$
|\psi(\theta)\rangle.
$$

The values in $\theta$ are rotation angles that the classical optimizer adjusts.

They are **not** Hamiltonian coefficients.

---

### `openfermion`

OpenFermion represents fermionic/qubit Hamiltonians and supports the mappings used by the original AIM code.

The Hamiltonian may contain Pauli strings such as:

$$
X_0 Z_1 X_2,
\qquad
Y_0 Z_1 Y_2,
\qquad
Z_0,
$$

with numerical coefficients.

Those terms together define the qubit Hamiltonian.

---

### `qiskit_port.dmft_qiskit`

This confirms that Python can see the Qiskit port itself.

If this import fails while Qiskit imports correctly, the problem is often one of these:

```text
wrong working directory
qiskit_port folder missing
repository layout changed
Python path issue
```

---

# 6. Optional but recommended: check Qiskit Aer

Run:

```bash
python -c "import qiskit_aer; print('Qiskit Aer: OK')"
```

If installed:

```text
Qiskit Aer: OK
```

## What Aer does

A quantum circuit does not automatically produce a statevector. The circuit must be simulated.

The Qiskit port can use:

```text
AerSimulator(method="statevector")
```

to simulate the circuit.

If Aer is unavailable, the code can fall back to:

```python
Statevector.from_instruction(circuit)
```

Both produce the same mathematical type of result: a complex statevector representing the amplitudes of the computational basis states.

Aer is recommended because repeated statevector generation is one of the largest runtime costs in this workflow.

The project therefore caches parameterized circuit templates and uses Aer when possible to reduce repeated overhead.

---

# 7. Minute 2–3: syntax-check the Qiskit files

Run:

```bash
python -m py_compile \
    qiskit_port/qiskit_utils.py \
    qiskit_port/qiskit_vqe.py \
    qiskit_port/dmft_qiskit.py
```

## What `py_compile` actually checks

Python reads each file and attempts to compile it into Python bytecode.

This catches errors such as:

```text
bad indentation
missing parentheses
invalid syntax
malformed function definitions
```

It does **not** prove the numerical algorithm is correct.

So:

```text
py_compile passes
```

means:

> Python can parse the files.

It does **not** mean:

> The Qiskit and Qulacs physics agree.

That is why the next step is a numerical smoke test.

If the command prints nothing and returns to the shell prompt, that normally means the syntax check passed.

---

# 8. Minute 3–8: run the smoke test

Run:

```bash
python smoke_test_qiskit.py
```

The smoke test uses:

```python
SYSTEM_SIZE = 4
SEED = 0
TARGET_ERR = 1.0
```

and calls:

```python
dmft_qiskit.run_gs_error_experiment_qiskit(...)
```

with the Green's-function stage disabled.

The exact call is conceptually:

```python
results = dmft_qiskit.run_gs_error_experiment_qiskit(
    system_size=4,
    seed=0,
    target_err=1.0,
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
    optimizer_seed=0,
)
```

The remainder of this section explains **every important argument and every major operation that occurs inside this call**.

---

# 9. Smoke-test parameters: what every value means

## `system_size=4`

This creates the four-site AIM used by the validated smoke test.

Under the repository convention:

```text
1 impurity site
3 bath sites
4 total spatial sites
8 spin orbitals
8 qubits
```

The quantum statevector therefore has

$$
2^8 = 256
$$

complex amplitudes.

---

## `seed=0`

This is the **AIM model seed**.

It controls the deterministic random parameters used when constructing the test AIM.

The important consequence is:

```text
same system_size + same seed
    -> same model Hamiltonian
```

Changing `seed` means you are no longer testing exactly the same physical model.

Do not confuse this with `optimizer_seed`.

---

## `optimizer_seed=0`

This controls the random starting parameters used by the variational optimizer.

The ansatz starts from a vector of circuit angles:

$$
\theta_0.
$$

Because optimization can follow different paths from different starting points, fixing this seed improves reproducibility.

The two seeds therefore have different jobs:

```text
seed
    controls the AIM Hamiltonian

optimizer_seed
    controls variational starting angles
```

---

## `starting_depth=1`

The first ansatz depth attempted is one layer.

A **layer** is one repetition of the parameterized ansatz gate pattern.

More layers give the circuit more adjustable parameters and therefore more expressive power.

For the validated $N=4$ model:

```text
L=1 -> 17 parameters
L=2 -> 34 parameters
L=3 -> 51 parameters
L=4 -> 68 parameters
```

The general implementation uses:

$$
N_{\mathrm{params}}
=
N_{\mathrm{layers}}
\left(
N_{\mathrm{edges}}
+
N_{\mathrm{qubits}}
\right).
$$

For $N=4$:

```text
n_qubits = 8
n_edges  = 9

parameters per layer = 8 + 9 = 17
```

---

## `pre_empt_layers=1`

This is the maximum depth the smoke test is allowed to try.

Since:

```text
starting_depth = 1
pre_empt_layers = 1
```

the smoke test can only try:

```text
L = 1
```

This is intentional.

The production workflow may search $L=1,2,3,4,\ldots$, but a smoke test should be short.

`pre_empt_layers` should be interpreted as:

> **maximum depth to search before stopping**

not:

> **the depth the physics requires**

---

## `target_err=1.0`

This is intentionally very loose.

The real validated production target may be:

```python
target_err = 1e-4
```

but the smoke test uses:

```python
target_err = 1.0
```

because the purpose is to make $L=1$ acceptable immediately.

The relevant ground-state error is:

$$
\epsilon_{\mathrm{GS}}
=
1-
\left|
\langle
\psi_{\mathrm{VQE}}
|
\psi_{\mathrm{exact}}
\rangle
\right|.
$$

The validated $L=1$ error is about:

$$
2.145\times10^{-2},
$$

which is much smaller than 1.0.

Therefore the smoke test accepts the first layer and stops.

This is a **software test**, not an accuracy test.

---

## `optimizer="BFGS"`

BFGS is a classical numerical optimizer.

It sees the quantum circuit as a function of adjustable parameters:

$$
\theta
=
(\theta_1,\theta_2,\ldots,\theta_p).
$$

For each candidate $\theta$:

```text
build/bind circuit parameters
        |
        v
simulate the statevector
        |
        v
calculate energy
        |
        v
return energy to BFGS
```

BFGS then proposes another parameter vector in an attempt to lower the energy.

This repeats until the stopping condition is reached.

---

## `gs_gtol=5e-4`

`gs_gtol` is the gradient-norm tolerance used for the ground-state BFGS optimization.

Very roughly, the optimizer stops successfully when the estimated gradient becomes sufficiently small.

A smaller `gs_gtol` generally asks for tighter convergence and may require more work.

This value controls **optimizer convergence**.

It is different from:

```text
target_err
```

which controls **physics/ground-state accuracy relative to exact**.

So there are two separate questions:

```text
Did BFGS converge?
Did the converged state meet the desired overlap error?
```

A production depth is accepted only when the workflow's acceptance conditions are satisfied.

---

## `gs_maxiter=int(1e6)`

This is a large upper bound on the number of optimizer iterations allowed during the ground-state calculation.

It is a safety ceiling, not a request that the optimizer actually perform one million iterations.

If BFGS converges earlier, it stops earlier.

---

## `gf_gtol=5e-5`

This is the gradient tolerance intended for the later variational-Lanczos / Green's-function optimizations.

In the smoke test:

```python
gs=True
```

so the Green's-function calculation is never reached.

The argument remains present because it is part of the production function signature.

---

## `gf_maxiter=int(1e6)`

This is the Green's-function optimizer iteration ceiling.

Again, it is unused during the smoke test because:

```python
gs=True
```

stops the workflow after the ground state.

---

## `conv_tol=1e-6`

This is the general convergence tolerance passed into the variational Lanczos path.

It matters primarily during the Green's-function calculation.

It is included in the smoke-test function call so the call shape remains close to a real production run.

---

## `gs=True`

This is one of the most important smoke-test flags.

It means:

> Calculate and validate the ground state, then stop.

Therefore this:

```python
gs=True
```

prevents:

```text
phi-/phi+ construction
variational Lanczos
continued fractions
Green's-function calculation
spectral-function plotting
```

That is why the smoke test is much faster than a full calculation.

For a full production calculation use:

```python
gs=False
```

---

## `display=True`

Print progress and numerical information to the terminal.

This is useful during onboarding because the developer can see where the workflow is.

---

## `plot=False`

Do not create plots during the smoke test.

There is no need to plot anything merely to determine whether the ground-state path is healthy.

---

# 10. What happens internally when the smoke test runs

The most important part of onboarding is understanding that the single function call is actually coordinating many operations.

The internal sequence is approximately:

```text
run_gs_error_experiment_qiskit()
    |
    +--> initialize_system()
    |
    +--> exact.solve_exact_gs()
    |
    +--> calculate_ground_state_qiskit()
    |       |
    |       +--> qv.solve_vqe_qiskit()
    |       |
    |       +--> qv.construct_vqe_gs_qiskit()
    |
    +--> compare_ground_states_qiskit()
    |
    +--> test target_err
    |
    +--> return results
```

Each stage is explained below.

---

# 11. Operation 1: initialize the AIM

The Qiskit driver reuses the backend-independent system initialization from the original project.

Conceptually, initialization does the following:

```text
Choose system size and model seed
        |
        v
Generate AIM model parameters
        |
        v
Construct impurity + bath model
        |
        v
Create spin-up/spin-down indices
        |
        v
Create ansatz connectivity graphs
        |
        v
Construct qubit Hamiltonian
```

The output includes quantities such as:

```text
impurity_orbital
test_model
n_orbitals
up_qubit_indices
down_qubit_indices
connected_graphs
qubit_hamiltonian
```

## `impurity_orbital`

The qubit/spin-orbital index associated with the impurity operator used later for particle addition/removal.

---

## `up_qubit_indices` and `down_qubit_indices`

These tell the program which qubits belong to the spin-up and spin-down registers.

They are needed to enforce the particle-number and spin structure of the ansatz.

---

## `connected_graphs`

These graphs specify which qubits/sites are connected by the ansatz gate structure.

The ansatz does not place arbitrary gates between every possible pair. The graph structure encodes the intended AIM connectivity.

---

## `qubit_hamiltonian`

This is the AIM Hamiltonian after the fermionic problem has been mapped to qubit operators.

It can be written schematically as:

$$
H
=
\sum_j c_j P_j,
$$

where:

```text
c_j = numerical coefficient
P_j = Pauli string
```

A Pauli string may look like:

$$
X_0 Z_1 X_2
$$

or

$$
Z_0.
$$

The Hamiltonian is what determines the energy of a quantum state.

---

# 12. Operation 2: exact ground-state calculation

The workflow calls:

```python
exact.solve_exact_gs(test_model)
```

This is the classical reference calculation.

Exact diagonalization solves the matrix eigenvalue problem:

$$
H|\psi_n\rangle
=
E_n|\psi_n\rangle.
$$

The ground state is the eigenvector associated with the smallest eigenvalue:

$$
E_0
=
\min_n E_n.
$$

For the validated $N=4,\ seed=0$ model:

$$
E_{\mathrm{exact}}
\approx
-6.3424949883.
$$

The exact solver also identifies:

```text
charge sector
spin sector
whether the ground state is degenerate
```

---

# 13. What a charge/spin sector means

The full Hilbert space contains states with different particle numbers and spin configurations.

Instead of treating every possible state as one undifferentiated search space, the algorithm organizes states into symmetry sectors.

A **charge sector** specifies total particle number.

A **spin sector** specifies the spin imbalance used by the code.

For a given pair:

```text
charge
spin
```

the code can determine:

```text
n_up
n_down
```

and therefore construct the correct initial occupation pattern for the ansatz.

This makes the variational search physically structured instead of asking one circuit to freely mix states that should belong to different conserved sectors.

---

# 14. Operation 3: search for the Qiskit VQE ground state

The high-level function:

```python
calculate_ground_state_qiskit(...)
```

calls:

```python
qv.solve_vqe_qiskit(...)
```

The VQE search is not simply one optimization.

It searches the allowed charge/spin sectors and asks:

> Which sector contains the lowest variational energy?

For each allowed sector, the algorithm approximately does:

```text
Determine n_up and n_down
        |
        v
Construct initial occupation state
        |
        v
Build parameterized ansatz
        |
        v
Generate initial theta vector
        |
        v
Run BFGS energy minimization
        |
        v
Record optimized energy
```

After all relevant sectors are checked:

```text
choose sector with lowest optimized energy
```

That becomes the Qiskit VQE ground-state candidate.

---

# 15. Operation 4: construct the initial occupation state

Before applying variational gates, the circuit starts from a computational-basis occupation pattern.

The helper:

```python
get_initial_occupations_indices_qiskit(...)
```

determines which qubits should initially be occupied based on:

```text
n_up
n_down
```

The ansatz then starts from the correct particle-number/spin sector.

---

# 16. Operation 5: build the ansatz

The **ansatz** is the adjustable quantum circuit used to approximate a state.

Think of it as:

```text
fixed circuit structure
+
adjustable angles theta
```

The graph tells the code where the gates go.

The parameter vector tells the code how strongly the parameterized rotations are applied.

The optimizer changes:

$$
\theta_1,\theta_2,\ldots,\theta_p
$$

but the overall ansatz architecture remains fixed for a chosen number of layers.

Increasing the number of layers increases the number of adjustable degrees of freedom.

That often improves representational power, but it also makes the optimization problem larger.

---

# 17. Operation 6: simulate the statevector

For every trial parameter vector $\theta$, the circuit generates a quantum state:

$$
|\psi(\theta)\rangle.
$$

Because this project is running a noiseless simulation rather than executing on physical quantum hardware, the entire statevector is computed classically.

For $n$ qubits, the statevector contains:

$$
2^n
$$

complex amplitudes.

For $N=4$:

```text
8 qubits
2^8 = 256 amplitudes
```

This repeated statevector generation is one reason the Qiskit path can be computationally expensive.

To reduce overhead, the final port:

```text
caches parameterized ansatz templates
binds new parameter values instead of rebuilding everything
uses Aer when available
```

---

# 18. Operation 7: calculate the energy expectation value

For a trial state, the ground-state cost function calculates:

$$
E(\theta)
=
\langle
\psi(\theta)
|
H
|
\psi(\theta)
\rangle.
$$

This is called an **expectation value**.

In plain language:

> If the system is in the trial state $|\psi(\theta)\rangle$, what energy does that state have under Hamiltonian $H$?

The variational principle states that:

$$
E(\theta) \ge E_0
$$

for normalized trial states, where $E_0$ is the exact ground-state energy.

So the optimizer tries to push the trial energy downward toward the exact ground-state energy.

---

# 19. Operation 8: BFGS optimization

BFGS receives a function that maps:

$$
\theta
\rightarrow
E(\theta).
$$

It repeatedly proposes new parameter values.

Conceptually:

```text
theta_0
  |
  v
E(theta_0)
  |
  v
estimate search direction
  |
  v
theta_1
  |
  v
E(theta_1)
  |
  v
repeat
```

Eventually it returns an optimization result containing information such as:

```text
best parameter vector
best cost/energy
success flag
iteration count
function-evaluation count
termination message
```

The Qiskit workflow records whether the local VQE optimizer reports success.

---

# 20. Operation 9: reconstruct the winning ground state

After the lowest-energy sector and best angles are known, the code calls:

```python
construct_vqe_gs_qiskit(...)
```

This rebuilds the final ansatz using the optimized angles and generates the final statevector.

The result is stored in two useful forms:

```text
raw Qiskit/circuit ordering
OpenFermion/Python ordering
```

Why two forms exist is explained next.

---

# 21. Critical concept: statevector ordering

This was one of the most important issues found during the Qiskit/Qulacs validation.

Different software representations may index computational-basis amplitudes in different bit orders.

The raw Qiskit statevector and the matrix representation used by the OpenFermion/exact side must be aligned before operations such as:

$$
\langle\psi_{\mathrm{VQE}}|\psi_{\mathrm{exact}}\rangle
$$

or:

$$
H|\psi\rangle.
$$

The validated conversion helper is:

```python
qulacs_to_python_ordering_qiskit(
    state,
    n_qubits,
)
```

Internally, this performs the required bit-index reversal/permutation.

The practical rule is:

```text
Keep Qiskit Krylov states in raw circuit ordering while doing circuit-based variational work.

Convert to OpenFermion/Python ordering only when interacting with the matrix/exact-solver representation.
```

Do not casually remove this conversion because the arrays can have the same length and norm while representing amplitudes in different basis positions.

---

# 22. Operation 10: compare Qiskit and exact ground states

The workflow computes two main quantities.

## Energy relative error

$$
\epsilon_E
=
\frac{
|E_{\mathrm{exact}}-E_{\mathrm{VQE}}|
}{
|E_{\mathrm{exact}}|
}.
$$

This asks:

> Relative to the exact energy scale, how far away is the VQE energy?

---

## Ground-state overlap

$$
O
=
\left|
\langle
\psi_{\mathrm{VQE}}
|
\psi_{\mathrm{exact}}
\rangle
\right|.
$$

If the two normalized states are identical up to a global phase:

$$
O=1.
$$

The workflow then defines:

$$
\epsilon_{\mathrm{GS}}
=
1-O.
$$

Therefore:

```text
GS error = 0
```

means perfect state overlap.

A small GS error means the variational state closely matches the exact state.

Note that the overlap error used here is based on the **absolute overlap**, not the squared fidelity.

If squared fidelity is reported:

$$
F
=
|\langle\psi_{\mathrm{VQE}}|\psi_{\mathrm{exact}}\rangle|^2.
$$

These two quantities should not be confused.

---

# 23. Operation 11: depth acceptance logic

For a normal production search, the workflow tries:

```text
L = starting_depth
L = starting_depth + 1
L = starting_depth + 2
...
```

until either:

```text
GS error < target_err
```

or:

```text
L reaches pre_empt_layers
```

For the smoke test:

```text
starting_depth = 1
pre_empt_layers = 1
target_err = 1.0
```

so only $L=1$ is tested and it easily passes the intentionally loose threshold.

The result includes:

```text
selected_depth
overall_success
depth_history
ground_state
```

---

# 24. What `selected_depth` means

If:

```python
selected_depth == 4
```

that means:

> $L=4$ was the first ansatz depth in the search that satisfied the ground-state error criterion.

It does **not** mean:

```text
N=4 always requires L=4
```

and it does not mean:

```text
all larger systems require more layers in a simple one-to-one pattern
```

Required depth depends on:

```text
the model
the ansatz
the optimization path
the requested accuracy
```

---

# 25. What `overall_success` means

The production depth-search logic tracks both the accuracy target and optimizer status.

The important distinction is:

```text
accuracy condition
    -> did the state meet target_err?

optimizer condition
    -> did the numerical optimizer report success?
```

A state can numerically have a good overlap while the optimizer reports a formal convergence failure, or the optimizer can report success while the result is not accurate enough for the chosen target.

The workflow records these separately and uses them to decide whether it is safe to proceed.

---

# 26. Expected smoke-test output

The validated reference is approximately:

```text
system_size       = 4
seed              = 0
selected_depth    = 1
VQE energy        ~= -6.24852694
GS overlap error  ~= 2.1453e-02
overall_success   = True
```

The exact energy for this model is approximately:

```text
E_exact ~= -6.3424949883
```

The $L=1$ VQE energy is deliberately not extremely close to exact.

That is expected.

The smoke test is checking:

```text
same model
same code path
same rough optimized result
same state-comparison logic
successful optimizer behavior
```

not production-quality accuracy.

The smoke test allows small numerical tolerance around the validated reference so that tiny platform-dependent floating-point changes do not cause a false failure.

---

# 27. What a smoke-test failure means

A failed smoke test does not automatically mean the physics algorithm is broken.

Interpret the failure by stage.

| Failure | Likely meaning |
|---|---|
| `ModuleNotFoundError` | Environment, working directory, or package installation problem. |
| Syntax error / indentation error | Source file was edited incorrectly or wrong file version is present. |
| `selected_depth` is missing | Depth search did not reach the expected acceptance path. |
| `overall_success=False` | Optimizer did not formally converge under the current conditions. |
| Energy is very different | Wrong model seed, wrong code version, optimizer behavior changed, state/hamiltonian issue, or numerical regression. |
| GS overlap error is very different | State ordering, reconstructed state, optimizer result, or exact/VQE comparison may have changed. |

Do not immediately change the Green's-function code when a ground-state smoke test fails. Fix the earliest failing stage first.

---

# 28. Minute 8–10: open `aim_tutorial_qiskit.ipynb`

After the smoke test passes, open:

```text
aim_tutorial_qiskit.ipynb
```

This notebook was intentionally designed to mirror the original Qulacs tutorial as closely as possible.

The point is that a developer familiar with:

```text
aim_tutorial.ipynb
```

can move to:

```text
aim_tutorial_qiskit.ipynb
```

without learning a completely different notebook structure.

For the first run, stop after the ground-state sections.

Do not launch the full Green's-function calculation merely to verify that the notebook opens.

---

# 29. Production ground-state search

Once the smoke test is working, a realistic ground-state-only depth search is:

```python
import qiskit_port.dmft_qiskit as dmft_qiskit

results = dmft_qiskit.run_gs_error_experiment_qiskit(
    system_size=4,
    seed=0,
    target_err=1e-4,
    gf_maxiter=int(1e6),
    gs_maxiter=int(1e6),
    gf_gtol=5e-5,
    gs_gtol=5e-4,
    pre_empt_layers=4,
    starting_depth=1,
    gs=True,
    display=True,
    plot=False,
    optimizer="BFGS",
    conv_tol=1e-6,
    optimizer_seed=0,
)
```

For the validated $N=4,\ seed=0$ reference, the depth history was approximately:

| Depth | VQE energy | GS overlap error | Meets $10^{-4}$? |
|---:|---:|---:|:---|
| 1 | -6.24852694 | $2.1453\times10^{-2}$ | No |
| 2 | -6.34025229 | $2.1034\times10^{-4}$ | No |
| 3 | -6.34072163 | $1.2594\times10^{-4}$ | No |
| 4 | -6.34223741 | $1.7447\times10^{-5}$ | Yes |

So the first accepted depth was:

```text
L = 4
```

This is why the production search exists: the correct depth is not assumed in advance.

---

# 30. Full production calculation

To proceed from the accepted ground state into the Green's-function calculation, use:

```python
results = dmft_qiskit.run_gs_error_experiment_qiskit(
    system_size=4,
    seed=0,
    target_err=1e-4,
    gf_maxiter=int(1e6),
    gs_maxiter=int(1e6),
    gf_gtol=5e-5,
    gs_gtol=5e-4,
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

The single change:

```python
gs=False
```

allows the workflow to continue after the ground state has been accepted.

The following sections explain what happens next.

---

# 31. Green's-function operation 1: shift the Hamiltonian

For the Green's-function calculation, the Hamiltonian is referenced to the variational ground-state energy.

Conceptually:

$$
H'
=
H-E_{\mathrm{VQE}}I.
$$

The utility function constructs matrix representations of:

$$
H'
$$

and

$$
(H')^2.
$$

These matrices are used inside the variational Lanczos recurrence and cost function.

The shift makes the ground-state reference energy effectively zero in the Green's-function construction.

---

# 32. Green's-function operation 2: construct particle-removal and particle-addition states

Starting from the variational ground state:

$$
|\psi_0\rangle,
$$

the workflow constructs two initial Krylov states.

Particle removal:

$$
|\phi^-\rangle
=
c|\psi_0\rangle.
$$

Particle addition:

$$
|\phi^+\rangle
=
c^\dagger|\psi_0\rangle.
$$

Here:

```text
c
    annihilates/removes a particle

c†
    creates/adds a particle
```

The implementation applies the appropriate Jordan-Wigner parity structure when generating these states.

These two branches answer different physical questions:

```text
What excitations are accessible after removing a particle?
What excitations are accessible after adding a particle?
```

The full Green's function combines both.

---

# 33. Why the norms of phi-minus and phi-plus matter

Applying $c$ or $c^\dagger$ to the ground state does not necessarily produce a normalized state.

The code therefore records:

$$
\|\phi^-\|
$$

and

$$
\|\phi^+\|.
$$

Their squared norms become weights multiplying the branch continued fractions.

If a branch norm is effectively zero, that branch contributes essentially nothing to the Green's function.

---

# 34. Green's-function operation 3: determine the new charge/spin sectors

Removing or adding a particle changes the symmetry sector.

Therefore the removal and addition branches do not generally live in the same sector as the original ground state.

The code computes:

```text
charge_minus
spin_minus

charge_plus
spin_plus
```

and determines the ideal Krylov dimensions associated with those sectors.

This step is backend-independent combinatorics and is reused from the original project.

---

# 35. Green's-function operation 4: what a Krylov space is

Starting from a vector $|\phi\rangle$, the mathematical Krylov space is:

$$
\mathcal K_m(H,\phi)
=
\operatorname{span}
\{
|\phi\rangle,
H|\phi\rangle,
H^2|\phi\rangle,
\ldots,
H^{m-1}|\phi\rangle
\}.
$$

Instead of working with the entire Hamiltonian matrix directly, Lanczos constructs a smaller basis that captures the action of the Hamiltonian on the physically relevant starting state.

In an exact classical Lanczos calculation, these Krylov vectors can be generated directly through matrix operations.

In this project, the variational method approximates the required Krylov states using parameterized quantum circuits.

That is why the method is called **variational Lanczos**.

---

# 36. Green's-function operation 5: Lanczos coefficients

Lanczos converts the Hamiltonian into an effective tridiagonal representation characterized by coefficients:

$$
a_0,a_1,a_2,\ldots
$$

and:

$$
b_1,b_2,b_3,\ldots.
$$

Conceptually:

```text
a_i
    diagonal energy-like coefficient for Krylov state i

b_i
    coupling between neighboring Krylov states
```

These coefficients are sufficient to construct the Green's function through a continued fraction.

---

# 37. Green's-function operation 6: variationally approximate each Krylov state

For a new Krylov state $u_i$, the code does not simply multiply a full statevector by $H$ and accept that vector as the next quantum state.

Instead, it finds ansatz parameters whose circuit state best satisfies the Lanczos recurrence constraints.

This is another optimization problem.

Therefore a Green's-function run contains:

```text
ground-state optimization
+
multiple removal-branch optimizations
+
multiple addition-branch optimizations
```

This is the main reason it is much slower than the ground-state smoke test.

---

# 38. What the Lanczos cost function is trying to enforce

The cost function penalizes a trial state when it does not behave like the next desired Krylov/Lanczos state.

At a high level it enforces conditions involving:

```text
the Hamiltonian action on the previous Krylov vector
normalization
orthogonality to earlier Krylov vectors
Lanczos recurrence consistency
```

The exact formula is implemented in:

```python
lanczos_cost_function_qiskit(...)
```

The optimizer changes the ansatz parameters until this cost is minimized.

This produces the next variational Krylov state.

---

# 39. Why complex `b` values are handled carefully

The Lanczos recurrence mathematically contains square roots when computing off-diagonal coefficients.

Floating-point arithmetic can introduce tiny imaginary components such as:

$$
x + 10^{-16}i.
$$

The implementation removes only negligible numerical imaginary noise under a strict tolerance.

It does **not** replace the quantity by an absolute value or arbitrarily clip negative values before the square root.

Changing this behavior can alter the mathematics of the recurrence and was specifically avoided during validation.

---

# 40. Green's-function operation 7: continued fraction

Once the $a_i$ and $b_i$ coefficients are known, the branch Green's function is evaluated through a continued fraction of the form:

$$
G(z)
=
\frac{1}{
z-a_0-
\frac{b_1^2}{
z-a_1-
\frac{b_2^2}{
z-a_2-\cdots
}
}
}.
$$

The numerical helper is:

```python
continued_fraction_qiskit(...)
```

This operation is cheap compared with the variational optimizations because the expensive work has already gone into obtaining the Lanczos coefficients.

---

# 41. What the frequency array `w` means

If the user does not provide a custom frequency grid, the workflow uses approximately:

```python
w = np.linspace(-25, 25, 1000, dtype=np.complex128) + 0.1j
```

This means:

```text
real part
    1000 frequency points from -25 to +25

imaginary part
    +0.1 i broadening
```

Mathematically:

$$
z
=
\omega+i\eta,
$$

where:

```text
omega
    real frequency

eta
    positive broadening parameter
```

The broadening prevents singular delta-like structures from appearing as infinitely sharp poles in the numerical plot.

---

# 42. Critical Green's-function sign convention

The validated implementation uses:

```text
particle removal
    evaluate the continued fraction at -w

particle addition
    evaluate the continued fraction at +w
```

and combines the branches as:

$$
G(\omega)
=
G_+(\omega)
-
G_-(\omega).
$$

This sign handling lives inside the Qiskit Green's-function implementation.

Do not copy the branch logic into a notebook and apply another sign manually.

A duplicated sign correction can produce a qualitatively wrong spectrum even when the individual branch calculations are correct.

---

# 43. Independent removal/addition random seeds

The final Qiskit implementation gives the removal and addition branches separate deterministic seed streams.

Why?

Suppose the removal optimizer fails once and retries with a new random starting point.

If both branches consume random numbers from one shared stream, the retry changes the random starting point that the addition branch receives.

Then:

```text
removal behavior
```

would unintentionally change:

```text
addition starting conditions
```

even though the two branches should be independently reproducible.

Using:

```text
seed_minus
seed_plus
```

prevents that coupling.

---

# 44. Green's-function operation 8: construct the spectral function

After the total retarded Green's function is available:

$$
G(\omega),
$$

the spectral function is:

$$
A(\omega)
=
-\frac{1}{\pi}
\operatorname{Im}
G(\omega).
$$

The spectral function is often the easiest output to interpret visually.

Important spectral features include:

```text
peak positions
peak heights
peak shapes
relative spectral weight
```

During Qiskit/Qulacs validation, matching peak positions and heights provides an intuitive visual check in addition to numerical array differences.

---

# 45. Green's-function operation 9: compare Qiskit with exact

The code calculates a relative error:

$$
\epsilon_G
=
\frac{
\|G_{\mathrm{Qiskit}}-G_{\mathrm{exact}}\|_2
}{
\|G_{\mathrm{exact}}\|_2
}.
$$

This is different from the ground-state overlap error.

Do not mix these metrics.

```text
GS error
    compares quantum states

GF relative error
    compares complex Green's-function arrays
```

The workflow can also record:

```text
average pointwise difference
maximum pointwise difference
```

depending on the result path.

---

# 46. The three most important error/convergence quantities

A new developer should distinguish these immediately.

| Quantity | Meaning |
|---|---|
| `gs_gtol` | Numerical optimizer convergence tolerance for the ground-state BFGS search. |
| `target_err` | Accuracy threshold for accepting a VQE ground state relative to the exact state overlap. |
| `gf_gtol` | Numerical optimizer convergence tolerance for variational Lanczos / Green's-function optimizations. |

They answer different questions.

```text
gs_gtol:
    Has the ground-state optimizer numerically converged?

target_err:
    Is the resulting ground state accurate enough for the chosen acceptance criterion?

gf_gtol:
    Have the Green's-function Krylov optimizations converged tightly enough?
```

---

# 47. Fixed-depth versus automatic-depth runs

There are two important high-level entry points.

## Fixed depth

```python
dmft_qiskit.run_aim_qiskit(...)
```

Use this when you specifically want:

```text
L = a chosen value
```

Examples:

```text
backend comparison
controlled regression test
profiling one ansatz depth
reproducing a known L=4 result
```

---

## Automatic depth

```python
dmft_qiskit.run_gs_error_experiment_qiskit(...)
```

Use this when you want:

```text
start at L = starting_depth
increase L
stop at first depth satisfying target_err
```

This is normally the production-style workflow.

---

# 48. How to read the returned results dictionary

A typical automatic-depth result contains structures such as:

```python
results["system"]
results["exact_ground_state"]
results["depth_history"]
results["selected_depth"]
results["overall_success"]
results["ground_state"]
```

If the Green's function runs successfully, it also contains data such as:

```python
results["green_function"]
results["errors"]
```

Important examples:

```python
results["selected_depth"]
```

The first accepted ansatz depth.

```python
results["depth_history"]
```

A record of the attempted depths and their errors/optimizer status.

```python
results["ground_state"]["energy"]
```

The accepted variational ground-state energy.

```python
results["ground_state"]["angles"]
```

The optimized ansatz parameters.

```python
results["ground_state"]["comparison"]["gs_error"]
```

The state-overlap error.

```python
results["green_function"]["g_total"]
```

The Qiskit Green's-function array.

```python
results["green_function"]["spectral"]
```

The Qiskit spectral-function array.

```python
results["errors"]["g_rel_error"]
```

The relative Green's-function error compared with exact.

---

# 49. What the low-level utilities mean

The main utility functions in `qiskit_utils.py` are worth understanding.

## `expectation_value(state, Hmat)`

Computes:

$$
\langle\psi|H|\psi\rangle.
$$

Used for energies and related quantities.

---

## `transition_amplitude(left_state, Hmat, right_state)`

Computes:

$$
\langle L|H|R\rangle.
$$

This is a matrix element between two different states.

---

## `overlap(left_state, right_state)`

Computes:

$$
\langle L|R\rangle.
$$

Used to determine how similar two states are and to enforce orthogonality.

---

## `continued_fraction_qiskit(w, a, b)`

Uses Lanczos coefficients to evaluate the frequency-dependent continued fraction that produces a Green's-function contribution.

---

## `qulacs_to_python_ordering_qiskit(state, n_qubits)`

Applies the validated basis-index reordering required when moving between raw circuit ordering and OpenFermion/Python matrix ordering.

---

## `create_qiskit_hamiltonian_matrix(...)`

Converts the qubit Hamiltonian into the matrix forms needed by the Qiskit variational calculations, including the shifted Hamiltonian and its square.

---

## `get_initial_occupations_indices_qiskit(...)`

Determines which qubits should begin occupied for the requested charge/spin sector.

---

# 50. Why Qiskit can be slower than Qulacs here

Controlled tests showed that Qiskit and Qulacs can perform the same optimizer workload and produce numerically equivalent answers while still having different runtimes.

The main reason is the simulator execution path.

Qulacs is optimized around direct simulator-native state updates.

The original Qiskit implementation repeatedly generated statevectors through a higher-level circuit path.

Because BFGS may evaluate the cost function thousands of times, even a small per-state overhead becomes significant.

The final Qiskit port therefore includes optimizations such as:

```text
cache the parameterized circuit template
bind new theta values
use Aer when available
cache the bit-reversal ordering map
precompute fixed Hamiltonian work where possible
```

The important conclusion is:

> A runtime difference by itself does not imply a physics disagreement.

Always compare numerical outputs before diagnosing a correctness problem from speed alone.

---

# 51. Validated backend-equivalence checkpoints

The strongest Qiskit/Qulacs validation was performed under controlled conditions where both backends evaluated equivalent states/parameters.

Representative validated results include:

```text
ground-state energy agreement
    ~ machine precision

Qiskit-Qulacs state fidelity
    ~ 1

phi- and phi+ state agreement
    ~ machine precision

Lanczos a/b coefficient differences
    ~ 1e-14 to 1e-13 scale

final spectral-function difference
    ~ floating-point scale in controlled same-state tests
```

See:

```text
VALIDATION.md
```

for the recorded numerical values.

These controlled comparisons are more informative about backend equivalence than comparing two completely independent stochastic optimization runs.

---

# 52. If you make a code change: minimum regression order

After changing the implementation, test from the cheapest stage to the most expensive stage.

```text
1. Import check
2. py_compile
3. smoke_test_qiskit.py
4. fixed-depth ground-state comparison
5. automatic depth search
6. phi-/phi+ validation if relevant
7. removal Lanczos branch
8. addition Lanczos branch
9. full Green's function
10. Qiskit vs Qulacs controlled comparison
```

This ordering isolates bugs.

If step 3 fails, there is no reason to wait for step 9.

---

# 53. Troubleshooting: import failure

If:

```text
ModuleNotFoundError: qiskit_port
```

check:

```bash
pwd
```

and make sure you are at the repository root.

Then inspect:

```bash
ls qiskit_port
```

You should see the Qiskit source files.

If Qiskit itself cannot import, verify the active environment.

---

# 54. Troubleshooting: Qiskit Aer is missing

If:

```text
ModuleNotFoundError: qiskit_aer
```

the code can fall back to the standard Qiskit statevector path.

This may be slower but should not by itself change the intended mathematics.

If runtime matters, install/use the same tested environment that includes Aer.

---

# 55. Troubleshooting: energy looks correct but overlap is wrong

This is a warning sign for **state representation/order**.

Two vectors can have the correct norm and even produce plausible energies while being indexed in incompatible basis orderings.

Check:

```text
raw Qiskit state
vs.
OpenFermion/Python ordered state
```

and confirm the comparison uses:

```python
state_ordered
```

rather than the raw circuit-ordering state.

Read the state-ordering section in `DEVELOPER_NOTES.md` before changing anything.

---

# 56. Troubleshooting: Qiskit and Qulacs start matching, then optimizers diverge

This can happen without a backend correctness bug.

BFGS is path-dependent.

Tiny floating-point differences can eventually make the optimizer choose a different search direction.

A strong diagnostic is **cross evaluation**:

```text
evaluate Qiskit cost at Qulacs theta
evaluate Qulacs cost at Qiskit theta
```

If the two backends agree on the cost at the same $\theta$, then the backend mathematics is agreeing even if independent optimization trajectories diverge later.

This distinction was important during validation.

---

# 57. Troubleshooting: Green's-function shape looks mirrored or sign-flipped

Do not immediately edit the plotting code.

First verify the branch convention:

```text
removal -> -w
addition -> +w
total G = G_plus - G_minus
```

A duplicated sign change can produce a visibly incorrect spectrum.

The sign logic should live in the Green's-function implementation, not be manually repeated in calling notebooks.

---

# 58. Troubleshooting: Green's-function run takes a long time

This is normally caused by repeated variational Lanczos optimization, not plotting.

Remember:

```text
one Green's-function calculation
    may include many Krylov states

each Krylov state
    may require many BFGS cost evaluations

each cost evaluation
    requires statevector simulation
```

Before launching a long run, verify:

```text
smoke test passes
ground-state depth is sensible
system size is intended
pre_empt_layers is intended
maxiter values are intended
Aer is available if expected
```

For quick development, run:

```python
gs=True
```

until you actually need the Green's function.

---

# 59. Troubleshooting: target not reached by `pre_empt_layers`

This means:

```text
the search ceiling was reached before the requested GS overlap error was achieved
```

It does not prove that the method can never reach the target.

Inspect:

```python
results["depth_history"]
```

If the error is improving as depth increases, extending `pre_empt_layers` may be reasonable.

If the error is erratic or the optimizer repeatedly fails, increasing depth blindly may only increase runtime.

---

# 60. What not to change casually

The following areas were specifically validated and can silently break correctness if modified without regression tests:

```text
statevector ordering conversion
Green's-function removal/addition sign convention
complex Lanczos b handling
independent branch seeding
ansatz parameter ordering
charge/spin-sector construction
continued-fraction logic
```

Read:

```text
DEVELOPER_NOTES.md
VALIDATION.md
```

before altering these pieces.

---

# 61. Glossary

| Term | Meaning in this project |
|---|---|
| AIM | Anderson Impurity Model. One interacting impurity coupled to bath sites. |
| Hamiltonian $H$ | Operator defining the energy and dynamics of the model. |
| Ground state | Lowest-energy eigenstate of $H$. |
| Exact diagonalization | Classical calculation of exact eigenvalues/eigenvectors for manageable system sizes. |
| VQE | Variational Quantum Eigensolver. Optimizes a parameterized quantum state to minimize energy. |
| Ansatz | Parameterized circuit architecture used to represent trial states. |
| Layer | One repetition of the ansatz gate pattern. |
| $\theta$ | Vector of trainable circuit rotation parameters. |
| BFGS | Classical numerical optimizer used to adjust $\theta$. |
| Charge sector | Subspace with a specified total particle number. |
| Spin sector | Subspace with a specified spin imbalance. |
| Statevector | Complex amplitude vector representing a pure quantum state. |
| Expectation value | $\langle\psi|H|\psi\rangle$. |
| Overlap | $\langle\phi|\psi\rangle$, measuring state similarity. |
| Fidelity | Squared magnitude of overlap for pure states. |
| Krylov space | Span of $\phi,H\phi,H^2\phi,\ldots$. |
| Lanczos | Procedure that represents Hamiltonian action using tridiagonal coefficients $a_i,b_i$. |
| $\phi^-$ | Particle-removal starting state $c|\psi_0\rangle$. |
| $\phi^+$ | Particle-addition starting state $c^\dagger|\psi_0\rangle$. |
| Green's function | Frequency-dependent response function constructed from addition/removal branches. |
| Spectral function | $A(\omega)=-\operatorname{Im}G(\omega)/\pi$. |
| `target_err` | Maximum accepted ground-state overlap error. |
| `gs_gtol` | Ground-state BFGS gradient tolerance. |
| `gf_gtol` | Variational-Lanczos BFGS gradient tolerance. |
| `starting_depth` | First ansatz depth tested. |
| `pre_empt_layers` | Maximum ansatz depth tested. |
| `optimizer_seed` | Seed controlling variational starting angles. |
| `seed_minus`, `seed_plus` | Independent random seeds for removal/addition optimization branches. |
| Aer | Qiskit high-performance simulator backend used for statevector simulation when available. |

---

# 62. The actual 10-minute checklist

After reading the explanations above once, the operational onboarding is short.

### Minute 0–2

```bash
conda activate aimenv
cd /path/to/qcaob-aim

python -c "import numpy, scipy, qiskit, openfermion; import qiskit_port.dmft_qiskit as d; print('Core imports: OK')"
python -c "import qiskit_aer; print('Qiskit Aer: OK')"
```

### Minute 2–3

```bash
python -m py_compile \
    qiskit_port/qiskit_utils.py \
    qiskit_port/qiskit_vqe.py \
    qiskit_port/dmft_qiskit.py
```

### Minute 3–8

```bash
python smoke_test_qiskit.py
```

Look for approximately:

```text
selected_depth   = 1
overall_success  = True
VQE energy       ~= -6.24852694
GS overlap error ~= 2.1453e-02
SMOKE TEST: PASS
```

### Minute 8–10

Open:

```text
aim_tutorial_qiskit.ipynb
```

Run through the ground-state section first.

If those steps work, the environment is ready for normal development.

---

# 63. What to read next

Use the files in this order:

```text
START_HERE.md
    -> understand and verify the environment

aim_tutorial_qiskit.ipynb
    -> see the workflow step by step

VALIDATION.md
    -> see known-good numerical references

DEVELOPER_NOTES.md
    -> understand why the implementation is structured this way

README.md
    -> use as the compact project-level reference
```

The key principle for future development is:

> **Change one layer of the workflow at a time and validate it before moving to the next.**

For this project, a correct ground state is the prerequisite for a meaningful Green's function, and a correct Green's function requires preserving the validated state-ordering, Lanczos, and addition/removal conventions.
