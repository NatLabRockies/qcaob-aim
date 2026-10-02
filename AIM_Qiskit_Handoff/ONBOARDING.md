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
