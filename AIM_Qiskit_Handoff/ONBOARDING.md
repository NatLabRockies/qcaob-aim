# ONBOARDING.md

This file is retained for compatibility with older links and handoff materials.

The current onboarding guide is:

[`START_HERE.md`](START_HERE.md)

Use `START_HERE.md` for:

- environment setup,
- repository structure,
- backend selection,
- the shared `dmft.py` workflow,
- Qiskit and Qulacs execution,
- the smoke test,
- ground-state VQE,
- variational Lanczos,
- Green's-function calculations,
- troubleshooting,
- and current regression guidance.

The project now uses one shared high-level workflow:

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
```

Historical development and debugging details are recorded in:

[`DEVELOPER_NOTES.md`](DEVELOPER_NOTES.md)
