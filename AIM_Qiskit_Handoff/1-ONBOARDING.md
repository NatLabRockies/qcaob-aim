# Qiskit Onboarding

Use this file as the quick-start path for a new developer working with the
Qiskit backend.

## 1. Set up the base repository environment

Start with the repository's main:

[`README.md`](README.md)

That file contains the original local Python/Pipenv setup and the base
dependencies used by the AIM repository.

## 2. Read the Qiskit technical guide

Then read:

[`README_QISKIT.md`](README_QISKIT.md)

That document contains the current Qiskit-specific setup, backend selection,
source-file layout, run commands, smoke test, VQE/Green's-function workflow,
important conventions, troubleshooting, and regression references.

## 3. Run the Qiskit smoke test

From the repository root:

```bash
python3 -m pipenv run python qiskit_port/smoke_test_qiskit.py
```

The validated small test uses the shared `dmft.py` workflow with:

```text
backend          = qiskit
system_size      = 2
seed             = 0
VQE energy       ~= -5.063959182148
GS overlap error ~= 5.64738e-05
```

and should finish with:

```text
PASSED
```

This is a fast environment/regression check. It does not run the full
variational-Lanczos Green's-function workflow.

## 4. Open the executable tutorial

After the smoke test passes, open:

```text
aim_tutorial_merged.ipynb
```

## 5. Use the developer notes when changing internals

For debugging history, validation evidence, numerical regression values,
performance investigations, and known implementation pitfalls, read:

[`DEVELOPER_NOTES.md`](DEVELOPER_NOTES.md)


## 6. IBM Quantum hardware workflow

After the local Qiskit smoke test and tutorial work correctly, use
`hardware_workflow.py` for the IBM hardware path.

For the full IBM-specific guide, read:

[`README_IBM_QUANTUM_WORKFLOW.md`](README_IBM_QUANTUM_WORKFLOW.md)

The current production strategy is:

```text
Ground state:
    Aer finite-shot VQE + SPSA
        ->
    optimized ground-state circuit
        ->
    IBM finite-shot energy measurement

Green's function:
    local variational-Lanczos / BFGS Krylov optimization
        ->
    optimized Krylov circuits
        ->
    IBM finite-shot measurement of GF coefficients
        ->
    classical continued-fraction reconstruction
```

The IBM QPU does **not** optimize the Green's-function Krylov states in the
current production workflow. The Krylov basis is optimized locally, and IBM is
used to measure the expectation values that determine the final Green's-function
coefficients.

### IBM and local Python environments

The workflow intentionally uses two Python environments.

Local AIM/Qiskit environment:

```text
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
```

IBM Runtime environment:

```text
~/venvs/qcaob-ibm-hardware/bin/python
```

Normally, run the public controller from the repository root with:

```bash
python3 hardware_workflow.py ...
```

The controller dispatches local and IBM work to the appropriate environment.

### Ground-state-only workflow

To optimize and export the ground state locally with finite-shot Aer + SPSA:

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

This does **not** submit an IBM job.

Important outputs:

```text
aim_hw_data.json
aim_hw_circuit.qasm
```

To optimize locally and then submit the ground-state circuit to IBM:

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

**This submits a real IBM QPU job and consumes runtime.**

If the optimized ground-state circuit already exists and you only want to submit
that saved circuit:

```bash
python3 hardware_workflow.py \
  --submit-only \
  --backend ibm_fez \
  --shots 5000
```

`--submit-only` also submits a real IBM QPU job.

After a ground-state job is submitted, record the returned Job ID and retrieve
that same job with:

```bash
python3 hardware_workflow.py \
  --retrieve JOB_ID
```

Retrieval does **not** submit another job.

### Green's-function hardware workflow

First optimize and export the Krylov basis locally:

```bash
python3 hardware_workflow.py \
  --gf-hardware-prepare
```

This does **not** submit an IBM job.

Before every GF hardware submission, run the dry check:

```bash
python3 hardware_workflow.py \
  --gf-hardware-dry-run
```

For the validated size-2 system, expect:

```text
Removal sector: 1 additional Krylov state
Addition sector: 1 additional Krylov state
Estimator PUBs: 10
IBM submission: NO
```

The corrected Krylov counting rule is:

```text
additional Krylov states = D - 1
```

because the initial state `u0` already exists.

For size 2:

```text
D_minus = 2 -> 1 additional removal state
D_plus  = 2 -> 1 additional addition state
```

To submit the prepared GF measurements to IBM:

```bash
python3 hardware_workflow.py \
  --gf-hardware-submit \
  --backend ibm_fez \
  --shots 5000
```

**This submits a real IBM QPU job and consumes runtime.**

Once a Job ID is returned, do not submit the same experiment again merely to
check its status.

Retrieve the existing GF job:

```bash
python3 hardware_workflow.py \
  --gf-hardware-retrieve JOB_ID
```

Then reconstruct and plot that exact job:

```bash
python3 hardware_workflow.py \
  --gf-hardware-plot JOB_ID
```

Retrieval and plotting do **not** submit new IBM jobs.

### Multiple IBM jobs and shot counts

Different IBM jobs may intentionally use different shot counts, such as:

```text
500
1,000
5,000
10,000
```

Always treat the IBM Job ID as the primary identifier for a run.

For new GF jobs, the workflow stores job-specific metadata, measurements,
results, and plots so that one job's shot count does not overwrite another.

Example job-specific files:

```text
aim_hw_gf_submission_JOBID.json
aim_hw_gf_measurements_JOBID.json
aim_hw_gf_results_JOBID.json
ibm_gf_combined_JOBID_5000shots.png
```

For older legacy jobs that were submitted before per-job metadata was saved,
the actual historical shot count may need to be supplied once during retrieval:

```bash
python3 hardware_workflow.py \
  --gf-hardware-retrieve JOB_ID \
  --job-shots 5000
```

Use the actual shot count used by that specific historical job.

### IBM command safety summary

| Command | Purpose | New IBM QPU job? |
|---|---|---:|
| `--prepare-only` | Local GS optimization/export | No |
| `--ground-state` | Local GS optimization + IBM GS measurement | **Yes** |
| `--submit-only` | Submit saved GS circuit | **Yes** |
| `--retrieve JOB_ID` | Retrieve GS result | No |
| `--gf-hardware-prepare` | Local GF/Krylov optimization/export | No |
| `--gf-hardware-dry-run` | Validate GF hardware package | No |
| `--gf-hardware-submit` | Submit IBM GF measurements | **Yes** |
| `--gf-hardware-retrieve JOB_ID` | Retrieve GF result | No |
| `--gf-hardware-plot JOB_ID` | Reconstruct/plot GF result | No |

The operational rule is:

> **If a real IBM submission returns a Job ID, keep that Job ID and retrieve the
> existing job. Do not submit a duplicate job just to check the result.**


## Current architecture

```text
                         dmft.py
                            |
                    BACKEND selector
                     /            \
                "qulacs"         "qiskit"
                    |               |
                  vqe.py      qiskit_port/
                                  |
                           qiskit_vqe.py
                                  |
                           qiskit_ansatz.py
                                  |
                           qiskit_utils.py
```

The former duplicated high-level Qiskit driver has been retired.

For Qiskit:

```python
import dmft
dmft.BACKEND = "qiskit"
```

For Qulacs:

```python
import dmft
dmft.BACKEND = "qulacs"
```
