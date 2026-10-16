# IBM Quantum Hardware Workflow for AIM/DMFT Ground States and Green's Functions

## Purpose

This document describes the **production IBM Quantum workflow** used in this repository for the Anderson Impurity Model (AIM) / Dynamical Mean-Field Theory (DMFT) Green's-function calculation.

The chosen workflow is:

> **Local optimization + IBM QPU measurement of Green's-function coefficients**

This is intentionally different from the earlier fully shot-only Krylov experiments. The local optimizer is used to obtain a stable variational Krylov basis, while the IBM QPU is used to measure the expectation values that determine the final Green's-function coefficients.

The main controller is:

```text
hardware_workflow.py
```

Run commands from the repository root.

---

# 1. Scientific workflow

The complete workflow is:

```text
AIM model
   |
   v
Local finite-shot ground-state VQE
Aer + SPSA
   |
   v
Local variational Krylov / Lanczos optimization
BFGS + statevector evaluation
   |
   v
Export optimized circuits and observables
   |
   v
IBM QPU measures GF observables
finite shots
   |
   v
Retrieve expectation values
   |
   v
Reconstruct Lanczos coefficients
a_i and b_i
   |
   v
Classical continued fraction
   |
   v
G^ret(omega) and A(omega)
   |
   v
Compare IBM result against exact reference
```

The IBM QPU **does not optimize the Krylov states** in this production workflow.

It measures the quantities required to reconstruct the Green's function after the variational basis has already been optimized locally.

---

# 2. Why this workflow is used

Three workflows were considered:

| Workflow | Accuracy / stability | QPU cost | Use |
|---|---:|---:|---|
| Fully local calculation | Highest stability | None | Reference / debugging |
| Fully shot-only Krylov optimization | Lower stability | Potentially high | Experimental |
| **Local optimization + IBM GF measurement** | **Better** | **Moderate** | **Production workflow** |

The production workflow keeps the difficult nonlinear variational optimization local and uses the IBM hardware for the physically relevant expectation-value measurements.

This reduces unnecessary QPU runtime while still producing a hardware-evaluated Green's function.

---

# 3. Required repository files

Keep these core files:

```text
hardware_workflow.py
dmft.py
exact.py
vqe.py
qiskit_port/
```

The workflow also generates files such as:

```text
aim_hw_data.json
aim_hw_circuit.qasm

aim_hw_gf_data.json
aim_hw_gf_circuits/

aim_hw_gf_submission.json
aim_hw_gf_measurements.json
aim_hw_gf_results.json
```

For new IBM jobs, the workflow also creates **job-specific files** so multiple jobs with different shot counts do not overwrite one another.

Examples:

```text
aim_hw_gf_submission_JOBID.json
aim_hw_gf_measurements_JOBID.json
aim_hw_gf_results_JOBID.json
ibm_gf_combined_JOBID_5000shots.png
```

The non-job-specific files remain convenience copies for the latest run.

---

# 4. Python environments

This project intentionally uses two Python environments.

## Local AIM environment

Used for:

- OpenFermion
- Qiskit Aer
- local VQE
- SPSA ground-state optimization
- local BFGS / statevector Krylov optimization
- plotting
- exact comparison

Configured path:

```text
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
```

## IBM Runtime environment

Used for:

- IBM Quantum authentication
- backend access
- transpilation against a real IBM backend
- IBM Runtime Estimator jobs
- job retrieval

Configured path:

```text
~/venvs/qcaob-ibm-hardware/bin/python
```

The controller automatically dispatches work to the appropriate Python environment.

Therefore, normally run:

```bash
python3 hardware_workflow.py ...
```

from the repository root.

Do **not** manually run the entire local AIM workflow from the IBM environment. The IBM environment does not need OpenFermion or Qiskit Aer for QPU submission/retrieval.

---

# 5. First-time IBM Quantum setup

Complete this section **once before the first IBM hardware run**. None of the setup or verification commands in this section submit a QPU job.

The current IBM Quantum Compute client uses the `ibm_quantum_platform` channel. The project therefore authenticates with:

```python
QiskitRuntimeService(
    channel="ibm_quantum_platform"
)
```

## 5.1 Create or sign in to an IBM account

1. Open the IBM Quantum Platform.
2. Sign in with an existing IBM account or create one.
3. Complete the IBM Cloud / IBM Quantum account setup requested by the platform.

Official entry point:

- [IBM Quantum Platform](https://quantum.cloud.ibm.com/)
- [IBM documentation — Set up your IBM Cloud account](https://quantum.cloud.ibm.com/docs/en/guides/cloud-setup)

If you were invited to an organization's IBM Cloud account, use the account or project supplied by that organization rather than creating a separate paid resource unnecessarily.

## 5.2 Create or select an IBM Quantum Compute instance

IBM hardware access is associated with a **Quantum Compute service instance**.

On the IBM Quantum Platform, open the **Instances** page and either:

- use an existing instance that already gives you QPU access, or
- create an instance under the plan available to your account.

IBM currently offers an **Open** plan as well as paid plans. The exact QPUs and amount of runtime available depend on the instance and plan attached to the account.

Each instance has a Cloud Resource Name (**CRN**). Copy the CRN if you want this project to use one specific instance consistently.

Official references:

- [IBM documentation — Create and manage instances](https://quantum.cloud.ibm.com/docs/en/guides/instances)
- [IBM documentation — Plans overview](https://quantum.cloud.ibm.com/docs/en/guides/plans-overview)

For this project, specifying an instance is recommended when more than one instance is attached to the account. Newer `qiskit-ibm-runtime` versions can also automatically select an accessible instance when none is explicitly saved.

## 5.3 Create an IBM Cloud API key

From the IBM Quantum Platform dashboard, create an **API key** and copy it to a secure location.

Treat the API key like a password.

Do **not**:

```text
put the API key in hardware_workflow.py
commit it to Git
drop it into a README
paste it into a public issue or notebook
share it in chat
```

The workflow does not need the token hard-coded in the repository. Save it once through `QiskitRuntimeService.save_account()` instead.

Official reference:

- [IBM documentation — Save your login credentials](https://quantum.cloud.ibm.com/docs/en/guides/save-credentials)

## 5.4 Create the dedicated IBM Runtime Python environment

This repository intentionally keeps IBM Runtime separate from the older local AIM/Qiskit environment.

If the IBM environment does not already exist, create it:

```bash
python3 -m venv ~/venvs/qcaob-ibm-hardware
source ~/venvs/qcaob-ibm-hardware/bin/activate
python -m pip install --upgrade pip
```

For the environment used while developing this workflow, the known working package versions are:

```text
Python:                  3.11.x
Qiskit:                  2.5.2
qiskit-ibm-runtime:      0.50.0
truststore:              installed
```

To reproduce that environment closely:

```bash
python -m pip install \
  "qiskit==2.5.2" \
  "qiskit-ibm-runtime==0.50.0" \
  truststore
```

IBM's general installation command for the current client is:

```bash
python -m pip install qiskit-ibm-runtime
```

Official reference:

- [IBM documentation — Install the IBM Quantum Compute client](https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit-runtime)

After installation, verify the environment:

```bash
~/venvs/qcaob-ibm-hardware/bin/python - <<'PY'
import qiskit
import qiskit_ibm_runtime
import truststore

print("Qiskit:", qiskit.__version__)
print("qiskit-ibm-runtime:", qiskit_ibm_runtime.__version__)
print("IBM environment imports: OK")
PY
```

This does **not** contact a QPU.

## 5.5 Save the IBM credentials locally

Run this once from the IBM environment.

Using `getpass` prevents the API key from being echoed on screen or stored directly in shell history:

```bash
~/venvs/qcaob-ibm-hardware/bin/python - <<'PY'
from getpass import getpass
from qiskit_ibm_runtime import QiskitRuntimeService

api_key = getpass("IBM Cloud API key: ")
instance = input(
    "IBM Quantum instance CRN (press Enter to allow automatic selection): "
).strip()

kwargs = {
    "channel": "ibm_quantum_platform",
    "token": api_key,
    "set_as_default": True,
}

if instance:
    kwargs["instance"] = instance

QiskitRuntimeService.save_account(**kwargs)
print("IBM Quantum credentials saved locally.")
PY
```

If a saved account already exists and you intentionally want to replace it, use `overwrite=True` in the `save_account(...)` call.

After credentials are saved, normal project code should **not** contain the API key. The workflow can simply create the service with:

```python
service = QiskitRuntimeService(
    channel="ibm_quantum_platform"
)
```

## 5.6 Verify authentication before using runtime

The project injects the operating system certificate store with `truststore` before connecting:

```python
import truststore
truststore.inject_into_ssl()
```

Run this safe connection test:

```bash
~/venvs/qcaob-ibm-hardware/bin/python - <<'PY'
import truststore
truststore.inject_into_ssl()

from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(
    channel="ibm_quantum_platform"
)

print("IBM connection successful")
print("Available backends:")
for backend in service.backends():
    print(" ", backend.name)
PY
```

This checks:

```text
saved credentials
IBM account access
instance access
network/TLS connection
visible IBM backends
```

It does **not** submit a QPU job.

Backend availability can change. If `ibm_fez` is not listed, do not assume the account is broken; first inspect the backends available to the selected instance.

## 5.7 Optional: verify the exact configured instance

If several IBM Quantum Compute instances are attached to the account, save or explicitly initialize the desired instance by CRN.

Example:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="YOUR_INSTANCE_CRN",
)
```

Do not commit the CRN if the repository is public unless you have confirmed that doing so is acceptable for the project. It is not an authentication secret like the API key, but it is still account-specific infrastructure metadata.

## 5.8 First-time setup checklist

Before running `--ground-state`, `--submit-only`, or `--gf-hardware-submit`, confirm:

```text
[ ] IBM account created and verified
[ ] IBM Quantum Compute instance available
[ ] API key created and stored securely
[ ] ~/venvs/qcaob-ibm-hardware exists
[ ] qiskit-ibm-runtime imports successfully
[ ] truststore imports successfully
[ ] credentials saved with QiskitRuntimeService.save_account()
[ ] QiskitRuntimeService connects successfully
[ ] desired backend appears in service.backends()
[ ] local AIM workflow has already been validated
```

Only after these checks should you move to commands that actually submit QPU work.

---

# 6. Default IBM settings

The current workflow defaults are:

```text
Backend:           ibm_fez
Shots:             5000
Resilience level:  1
Transpile seed:    1234
```

These can be overridden from the command line.

Example:

```bash
python3 hardware_workflow.py \
  --gf-hardware-submit \
  --backend ibm_fez \
  --shots 5000
```

---

# 7. Step 1 — Prepare the ground state locally

This step performs the local ground-state calculation and exports the optimized circuit.

For the current size-2 test:

```bash
python3 hardware_workflow.py \
  --prepare-only \
  --size 2 \
  --depth 1 \
  --shots 5000 \
  --maxiters 500 \
  --vqe-seed 1 \
  --shot-seed 9102
```

This performs:

```text
Aer finite-shot VQE
+
SPSA optimization
```

and does **not** submit anything to IBM.

Important outputs include:

```text
aim_hw_data.json
aim_hw_circuit.qasm
```

The ground-state JSON stores the system information and optimized parameters required by later steps.

---

## 7.1 Run a ground-state-only IBM hardware job

The ground state can be run independently from the Green's-function workflow.

### Local ground-state preparation only — NO IBM submission

Use this to optimize the ground state with finite-shot Aer + SPSA and save the optimized circuit without using QPU runtime:

```bash
python3 hardware_workflow.py \
  --prepare-only \
  --size 2 \
  --depth 1 \
  --shots 5000 \
  --maxiters 500 \
  --vqe-seed 1 \
  --shot-seed 9102
```

This produces or updates:

```text
aim_hw_data.json
aim_hw_circuit.qasm
```

`--prepare-only` does **not** submit anything to IBM.

### Complete ground-state hardware run — REAL IBM QPU submission

Use this when you want to optimize/export the ground state locally and then measure the optimized ground-state circuit on IBM hardware:

```bash
python3 hardware_workflow.py \
  --ground-state \
  --size 2 \
  --depth 1 \
  --shots 5000 \
  --maxiters 500 \
  --vqe-seed 1 \
  --shot-seed 9102 \
  --backend ibm_fez
```

**WARNING: `--ground-state` submits a real IBM QPU job and consumes runtime.**

The ground-state hardware path is:

```text
Aer finite-shot VQE + SPSA
        |
        v
optimized ground-state circuit
        |
        v
IBM finite-shot measurement of <H>
```

The IBM ground-state job contains one Estimator PUB for the Hamiltonian expectation value. When submission succeeds, record the Job ID, backend, shot count, ISA depth, and ISA operation count.

The latest ground-state submission metadata is stored in:

```text
aim_hw_submission.json
```

### Submit an already-prepared ground state — REAL IBM QPU submission

If `aim_hw_data.json` and `aim_hw_circuit.qasm` already contain the optimized ground state you want to measure, submit that existing export without rerunning local VQE:

```bash
python3 hardware_workflow.py \
  --submit-only \
  --backend ibm_fez \
  --shots 5000
```

**WARNING: `--submit-only` also submits a real IBM QPU job.**

Use this when you intentionally want another hardware measurement of the same prepared circuit, for example at a different shot count.

## 7.2 Retrieve a ground-state IBM result

After a ground-state job has been submitted, do **not** submit another job just to check whether it finished. Retrieve the existing Job ID:

```bash
python3 hardware_workflow.py \
  --retrieve JOB_ID
```

Example:

```bash
python3 hardware_workflow.py \
  --retrieve dbxxxxxxxxxxxxxxxx
```

Retrieval does **not** submit a new QPU job.

When the job is `DONE`, the script reports:

```text
Reference statevector energy
IBM QPU ground-state energy
QPU uncertainty
Absolute difference
Relative difference
```

The retrieved hardware values are saved back into `aim_hw_data.json`.

The relative difference is

\[
\frac{|E_{\mathrm{IBM}}-E_{\mathrm{reference}}|}{|E_{\mathrm{reference}}|}.
\]

### Ground-state command summary

| Goal | Command | New IBM QPU job? |
|---|---|---:|
| Optimize/export GS locally | `python3 hardware_workflow.py --prepare-only ...` | No |
| Optimize locally + submit GS | `python3 hardware_workflow.py --ground-state ...` | **Yes** |
| Submit existing GS export | `python3 hardware_workflow.py --submit-only ...` | **Yes** |
| Retrieve GS job | `python3 hardware_workflow.py --retrieve JOB_ID` | No |

### Ground-state record keeping

`aim_hw_submission.json` is a convenience file for the latest ground-state submission. Always record the IBM Job ID when the job is submitted, especially if several shot-count experiments will be run.

---

# 8. Step 2 — Optimize the Green's-function Krylov basis locally

Run:

```bash
python3 hardware_workflow.py --gf-hardware-prepare
```

This step:

1. loads the saved ground-state solution;
2. constructs the particle-removal and particle-addition sectors;
3. performs local variational Lanczos / Krylov optimization;
4. stores the optimized circuit for every additional Krylov state;
5. exports all observables needed for IBM measurement;
6. stores local and exact Green's-function curves for later comparison.

No IBM QPU job is submitted.

The principal outputs are:

```text
aim_hw_gf_data.json
aim_hw_gf_circuits/
```

---

# 9. Krylov-state counting rule

The Krylov chain must use the corrected rule:

```text
number of additional Krylov states = D - 1
```

because the initial state \(u_0\) already exists.

## System size 2

For the current size-2 calculation:

```text
D_minus = 2
D_plus  = 2
```

Therefore:

```text
Removal additional states = 1
Addition additional states = 1
```

## Corrected size 3

For the previously used size-3 case:

```text
D_minus = 3
D_plus  = 9
```

Therefore:

```text
Removal additional states = 2
Addition additional states = 8
```

The workflow checks this before IBM submission.

If the exported state count does not agree with \(D-1\), do **not** submit the job.

---

# 10. Step 3 — Perform the safe IBM dry run

Before every real QPU submission, run:

```bash
python3 hardware_workflow.py --gf-hardware-dry-run
```

This performs validation only.

It does **not** connect to or submit a job to IBM.

For size 2, the expected output is approximately:

```text
Removal: D=2, additional states=1
Addition: D=2, additional states=1
Estimator PUBs required: 10
IBM backend: ibm_fez
Shots per PUB: 5000
IBM submission: NO
```

Do not proceed if the state counts or PUB count are unexpected.

---

# 11. Why size 2 requires 10 Estimator PUBs

The IBM GF job measures six observables using the ground-state circuit:

```text
gs.removal_weight
gs.addition_weight

gs.removal_H0
gs.removal_H20

gs.addition_H0
gs.addition_H20
```

That gives:

```text
6 PUBs
```

For every additional Krylov state, IBM measures:

```text
<H>
<H^2>
```

Size 2 has:

```text
1 removal Krylov state
1 addition Krylov state
```

so:

```text
2 states x 2 measurements = 4 PUBs
```

Total:

```text
6 + 4 = 10 PUBs
```

In general:

\[
N_{\mathrm{PUB}}
=
6
+
2\left(N_{\mathrm{Krylov},-}+N_{\mathrm{Krylov},+}\right)
\]

For the corrected size-3 calculation:

```text
6 + 2(2 + 8) = 26 PUBs
```

---

# 12. Step 4 — Submit the IBM Green's-function job

## WARNING

The following command submits a **real IBM QPU job** and consumes runtime:

```bash
python3 hardware_workflow.py \
  --gf-hardware-submit \
  --backend ibm_fez \
  --shots 5000
```

The workflow:

1. loads the locally optimized circuits;
2. loads \(H\) and \(H^2\);
3. obtains the selected IBM backend;
4. creates an optimization-level-3 preset pass manager;
5. transpiles each circuit to the backend ISA;
6. applies the final circuit layout to every observable;
7. creates all Estimator PUBs;
8. sets the Runtime resilience level;
9. sets the requested shot count;
10. submits one Estimator job containing all PUBs.

The code uses:

```python
estimator.options.resilience_level = args.resilience_level
estimator.options.default_shots = args.shots
```

Once a job has been submitted, record the returned:

```text
Job ID
Backend
Shots per PUB
Estimator PUB count
```

If a Job ID was returned, **do not submit again just because the job has not completed yet**.

Retrieve the existing job instead.

---

# 13. Multiple IBM jobs with different shot counts

The workflow supports multiple hardware jobs.

For example, one may intentionally run:

```text
500 shots
5,000 shots
20,000 shots
```

Each job has its own IBM Job ID.

The shot count displayed in a result or plot must correspond to the **selected IBM Job ID**, not to the current command-line default.

For newly submitted jobs, the script archives submission metadata under:

```text
aim_hw_gf_submission_JOBID.json
```

Example:

```text
aim_hw_gf_submission_dbxxxxxxxxxxxx.json
```

After retrieval it creates:

```text
aim_hw_gf_measurements_JOBID.json
```

and after plotting:

```text
aim_hw_gf_results_JOBID.json
ibm_gf_combined_JOBID_Nshots.png
```

This prevents a later 500-shot job from relabeling an earlier 5,000-shot result.

---

# 14. Step 5 — Retrieve an IBM GF job

Retrieval does **not** submit a new QPU job.

Use:

```bash
python3 hardware_workflow.py \
  --gf-hardware-retrieve JOB_ID
```

Example:

```bash
python3 hardware_workflow.py \
  --gf-hardware-retrieve db95vfs24mrc7383hab0
```

The workflow asks IBM Runtime for that exact job:

```python
job = service.job(JOB_ID)
```

If the job is not finished, the script prints the current status and returns.

If the job status is `DONE`, it extracts from every PUB:

```text
expectation value
standard deviation
```

and associates those values with the saved measurement labels.

For a newly submitted job, the labels and shot count are loaded from the per-job submission metadata.

---

# 15. Shot-count tracking

The result must use the shot count belonging to the specific IBM job.

The retrieval code attempts to determine the shot count in this order:

1. IBM Runtime job metadata;
2. IBM PUB/result metadata;
3. the archived per-job submission JSON;
4. a manual legacy override, only when necessary.

The final per-job measurement JSON records:

```json
{
  "job_id": "...",
  "backend": "...",
  "shots": 5000,
  "shots_source": "...",
  "measurements": {}
}
```

The plot reads the shot count from this retrieved per-job metadata.

It does **not** use the current `--shots` default to relabel an older job.

---

# 16. Retrieving legacy IBM jobs

Some IBM jobs were submitted before the repository saved immutable per-job metadata.

For those jobs, retrieval attempts to reconstruct the original PUB labels from:

```text
aim_hw_gf_data.json
```

The reconstructed order is:

```text
gs.removal_weight
gs.addition_weight
gs.removal_H0
gs.removal_H20
gs.addition_H0
gs.addition_H20

minus.u1.H
minus.u1.H2
...

plus.u1.H
plus.u1.H2
...
```

The script verifies that the number of IBM PUB results exactly matches the reconstructed label count.

If the counts do not agree, it stops rather than guessing.

---

# 17. Legacy job shot-count fallback

For an older IBM job, Runtime may no longer expose a directly recoverable shot count.

If that happens, retrieval will ask for:

```text
--job-shots N
```

Example:

```bash
python3 hardware_workflow.py \
  --gf-hardware-retrieve db95vfs24mrc7383hab0 \
  --job-shots 5000
```

Use this only when `5000` was actually the shot count used for that specific historical job.

This command is still retrieval-only.

It does **not** submit another QPU job.

Do not assign the same value to multiple legacy jobs unless they were actually submitted with the same number of shots.

---

# 18. Step 6 — Plot a specific IBM job

After retrieval, plot the exact job:

```bash
python3 hardware_workflow.py \
  --gf-hardware-plot JOB_ID
```

Example:

```bash
python3 hardware_workflow.py \
  --gf-hardware-plot db95vfs24mrc7383hab0
```

This plotting step is local.

It does **not** submit anything to IBM.

To plot the most recently retrieved job:

```bash
python3 hardware_workflow.py --gf-hardware-plot
```

For scientific reproducibility, using the explicit Job ID is preferable.

---

# 19. Final plot format

The final presentation figure is:

```text
ibm_gf_combined.png
```

and a job-specific copy is also saved:

```text
ibm_gf_combined_JOBID_Nshots.png
```

The plot uses:

```text
solid blue     IBM real Green's function
solid orange   IBM spectral function
dashed blue    exact real Green's function
dashed orange  exact spectral function
```

The spectral function is:

\[
A(\omega)
=
-\frac{1}{\pi}
\operatorname{Im}
G^{\mathrm{ret}}(\omega)
\]

The title follows the format:

```text
Qiskit | ibm_fez | Shots (5,000) | N_sys = 2 | n_layers = 1 | GF Rel. Err.=...
```

The shot count in the title comes from the selected IBM job's retrieved metadata.

---

# 20. Green's-function reconstruction

The IBM hardware does not directly return an entire \(G(\omega)\) curve.

Instead, it returns the expectation values used to calculate the Lanczos coefficients.

For each branch, the workflow reconstructs:

```text
weight
a_0, a_1, ...
b_1, b_2, ...
```

The two branches are:

```text
minus = particle removal
plus  = particle addition
```

The Green's function is then reconstructed classically through the continued fraction.

Conceptually:

\[
G(\omega)
=
G_+(\omega)
-
G_-(\omega)
\]

where the particle-addition and particle-removal contributions use their respective branch weights and Lanczos coefficients.

The final IBM result is compared with:

1. the exact Green's function;
2. the locally optimized Qiskit Green's function.

---

# 21. Result files

After plotting, the latest reconstructed result is saved as:

```text
aim_hw_gf_results.json
```

A permanent per-job result is also saved as:

```text
aim_hw_gf_results_JOBID.json
```

The result contains fields such as:

```text
job_id
backend
shots

weight_minus
weight_plus

a_minus
b_minus
a_plus
b_plus

g_ibm_real
g_ibm_imag

relative_error_vs_exact
relative_error_vs_local_qiskit
```

The per-job result is the preferred file for comparisons across different shot counts.

---

# 22. Recommended multi-shot experiment

To study convergence with shots, keep the local optimized basis fixed and submit separate IBM jobs.

For example:

## 500 shots

```bash
python3 hardware_workflow.py \
  --gf-hardware-submit \
  --backend ibm_fez \
  --shots 500
```

Record the returned Job ID.

## 5,000 shots

```bash
python3 hardware_workflow.py \
  --gf-hardware-submit \
  --backend ibm_fez \
  --shots 5000
```

Record the returned Job ID.

## 20,000 shots

```bash
python3 hardware_workflow.py \
  --gf-hardware-submit \
  --backend ibm_fez \
  --shots 20000
```

Record the returned Job ID.

Then retrieve each job independently:

```bash
python3 hardware_workflow.py --gf-hardware-retrieve JOB_ID_500
python3 hardware_workflow.py --gf-hardware-retrieve JOB_ID_5000
python3 hardware_workflow.py --gf-hardware-retrieve JOB_ID_20000
```

Then plot each independently:

```bash
python3 hardware_workflow.py --gf-hardware-plot JOB_ID_500
python3 hardware_workflow.py --gf-hardware-plot JOB_ID_5000
python3 hardware_workflow.py --gf-hardware-plot JOB_ID_20000
```

Do not change the local Krylov basis between these submissions if the purpose is to isolate the effect of shot count.

---

# 23. Important safety rule for IBM runtime

The following commands are safe and do **not** submit new QPU work:

```bash
python3 hardware_workflow.py --prepare-only ...
python3 hardware_workflow.py --gf-hardware-prepare
python3 hardware_workflow.py --gf-hardware-dry-run
python3 hardware_workflow.py --gf-hardware-retrieve JOB_ID
python3 hardware_workflow.py --gf-hardware-plot JOB_ID
```

The following commands **do submit real IBM QPU jobs**:

```bash
python3 hardware_workflow.py --gf-hardware-submit ...
python3 hardware_workflow.py --ground-state ...
python3 hardware_workflow.py --submit-only ...
```

Ground-state retrieval is safe:

```bash
python3 hardware_workflow.py --retrieve JOB_ID
```

and `--prepare-only` does not submit anything to IBM.

Always use the dry run before a GF submission.

---

# 24. Troubleshooting

## Error: `ModuleNotFoundError: No module named 'openfermion'`

This usually means the local scientific workflow was accidentally executed inside the IBM-only environment.

OpenFermion belongs to the local AIM environment.

Use the public controller:

```bash
python3 hardware_workflow.py ...
```

instead of manually forcing the full script through:

```text
~/venvs/qcaob-ibm-hardware/bin/python
```

The controller will launch only the IBM worker portions in the IBM environment.

---

## Error: job retrieval says there is no matching submission metadata

For a new job, verify that the per-job file exists:

```text
aim_hw_gf_submission_JOBID.json
```

For an older job, use the workflow version that supports legacy reconstruction.

The legacy retrieval path reconstructs labels from the current GF export and verifies the PUB count.

---

## Error: legacy job cannot determine the shot count

Use the historical shot count explicitly:

```bash
python3 hardware_workflow.py \
  --gf-hardware-retrieve JOB_ID \
  --job-shots N
```

This does not submit another job.

Only use the actual shot count originally used for that Job ID.

---

## Plot displays the wrong number of shots

Do not fix the title manually.

The correct workflow is:

```bash
python3 hardware_workflow.py \
  --gf-hardware-retrieve JOB_ID
```

followed by:

```bash
python3 hardware_workflow.py \
  --gf-hardware-plot JOB_ID
```

The plot should read the shot count from that job's retrieved metadata.

For newly submitted jobs, verify these files:

```text
aim_hw_gf_submission_JOBID.json
aim_hw_gf_measurements_JOBID.json
```

---

## Wrong PUB count

Run:

```bash
python3 hardware_workflow.py --gf-hardware-dry-run
```

For size 2, expect:

```text
Removal states: 1
Addition states: 1
Estimator PUBs: 10
```

If the count differs, do not submit until the exported Krylov dimensions are understood.

---

## Job status is not `DONE`

Simply retrieve again later:

```bash
python3 hardware_workflow.py \
  --gf-hardware-retrieve JOB_ID
```

Do not submit a duplicate job.

---

## Plot cannot find a job-specific measurement file

Retrieve the job first:

```bash
python3 hardware_workflow.py \
  --gf-hardware-retrieve JOB_ID
```

Then:

```bash
python3 hardware_workflow.py \
  --gf-hardware-plot JOB_ID
```

---

# 25. Typical complete size-2 session

```bash
# 1. Local finite-shot ground-state optimization.
# NO IBM submission.
python3 hardware_workflow.py \
  --prepare-only \
  --size 2 \
  --depth 1 \
  --shots 5000 \
  --maxiters 500 \
  --vqe-seed 1 \
  --shot-seed 9102

# 2. Local BFGS/statevector Krylov optimization.
# NO IBM submission.
python3 hardware_workflow.py \
  --gf-hardware-prepare

# 3. Validate state counts and QPU PUB count.
# NO IBM submission.
python3 hardware_workflow.py \
  --gf-hardware-dry-run

# 4. REAL IBM QPU SUBMISSION.
python3 hardware_workflow.py \
  --gf-hardware-submit \
  --backend ibm_fez \
  --shots 5000

# Record the Job ID printed here.

# 5. Retrieve the existing job.
# NO new submission.
python3 hardware_workflow.py \
  --gf-hardware-retrieve JOB_ID

# 6. Reconstruct and plot this exact job.
# NO IBM submission.
python3 hardware_workflow.py \
  --gf-hardware-plot JOB_ID
```

---

# 26. Recommended record keeping

For every IBM run, record at minimum:

| Field | Example |
|---|---|
| Job ID | `db...` |
| Backend | `ibm_fez` |
| Shots | `5000` |
| System size | `2` |
| VQE depth | `1` |
| Resilience level | `1` |
| Transpile seed | `1234` |
| Number of PUBs | `10` |
| GF relative error vs exact | result |
| GF relative error vs local Qiskit | result |
| Per-job result JSON | filename |
| Per-job plot | filename |

Do not identify runs only by shot count. The IBM **Job ID** is the primary identifier.

---

# 27. Reproducibility notes

When comparing multiple IBM jobs:

- use the same `aim_hw_gf_data.json`;
- use the same local optimized Krylov circuits;
- use the same backend when possible;
- use the same transpile seed when possible;
- change only the intended experimental variable, such as shot count;
- preserve every Job ID and job-specific result file;
- do not overwrite old measurement metadata with another job's shot count;
- compare both the numerical GF relative error and the full plotted curve.

Hardware results can still differ between jobs because of device calibration, gate error, readout error, transpilation, and finite sampling.

---

# 28. Current interpretation of the workflow

This calculation should be described as:

> The variational ground state and Krylov basis are optimized locally. The optimized circuits are then transpiled for IBM quantum hardware, where finite-shot expectation values are measured for the particle-addition and particle-removal weights and for the Hamiltonian moments required to reconstruct the Lanczos coefficients. The retarded Green's function is reconstructed classically from those QPU-measured coefficients and compared with an exact reference.

It should **not** be described as a fully hardware-optimized Krylov calculation.

It should also **not** be described as a statevector-only Green's-function result, because the final coefficients used for the IBM curve are obtained from finite-shot IBM QPU measurements.

---

# 29. Quick command reference

| Task | Command | QPU submission? |
|---|---|---:|
| Prepare ground state locally | `python3 hardware_workflow.py --prepare-only ...` | No |
| Run full IBM ground-state job | `python3 hardware_workflow.py --ground-state ... --backend ibm_fez --shots N` | **Yes** |
| Submit existing ground-state export | `python3 hardware_workflow.py --submit-only --backend ibm_fez --shots N` | **Yes** |
| Retrieve ground-state job | `python3 hardware_workflow.py --retrieve JOB_ID` | No |
| Prepare GF/Krylov basis | `python3 hardware_workflow.py --gf-hardware-prepare` | No |
| Validate PUB count | `python3 hardware_workflow.py --gf-hardware-dry-run` | No |
| Submit GF measurements | `python3 hardware_workflow.py --gf-hardware-submit --backend ibm_fez --shots N` | **Yes** |
| Retrieve job | `python3 hardware_workflow.py --gf-hardware-retrieve JOB_ID` | No |
| Retrieve legacy job with known shots | `python3 hardware_workflow.py --gf-hardware-retrieve JOB_ID --job-shots N` | No |
| Plot specific job | `python3 hardware_workflow.py --gf-hardware-plot JOB_ID` | No |
| Plot most recent retrieval | `python3 hardware_workflow.py --gf-hardware-plot` | No |

---

# 30. Final caution

A successful IBM submission returns a **Job ID**.

Once that ID exists:

```text
DO NOT resubmit merely to check the result.
```

Use:

```bash
python3 hardware_workflow.py \
  --gf-hardware-retrieve JOB_ID
```

until the job reaches `DONE`.

Only create another submission when a genuinely new hardware experiment is intended.
