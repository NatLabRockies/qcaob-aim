#!/usr/bin/env python3
"""HPC-friendly runner for the finished qcaob-aim repository on Kestrel."""

from __future__ import annotations

import argparse
import os
import platform
import socket
import sys
import time
from pathlib import Path

# Kestrel compute nodes are normally headless.
os.environ.setdefault("MPLBACKEND", "Agg")

# Respect Slurm CPU allocation and avoid accidental oversubscription.
_threads = os.environ.get("SLURM_CPUS_PER_TASK", "1")
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, _threads)

# This file should live in the repository root next to dmft.py.
REPO_ROOT = Path(__file__).resolve().parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import dmft  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the shared Qiskit/Qulacs qcaob-aim workflow on Kestrel."
    )
    p.add_argument("--backend", choices=("qiskit", "qulacs"), default="qiskit")
    p.add_argument("--system-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-error", type=float, default=1e-4)
    p.add_argument("--maxiters", type=int, default=1_000_000)
    p.add_argument("--gs-gtol", type=float, default=5e-4)
    p.add_argument("--gf-gtol", type=float, default=5e-5)
    p.add_argument("--starting-depth", type=int, default=1)
    p.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="Maximum VQE depth tested (dmft pre_empt_layers).",
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--ground-state-only",
        dest="ground_state_only",
        action="store_true",
        help="Stop after the ground-state/depth-search stage.",
    )
    mode.add_argument(
        "--full",
        dest="ground_state_only",
        action="store_false",
        help="Continue into the Green's-function workflow.",
    )
    p.set_defaults(ground_state_only=True)

    p.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return p


def validate(args: argparse.Namespace) -> None:
    if args.system_size < 2:
        raise ValueError("--system-size must be at least 2")
    if args.starting_depth < 1 or args.max_depth < 1:
        raise ValueError("VQE depths must be at least 1")
    if args.starting_depth > args.max_depth:
        raise ValueError("--starting-depth cannot exceed --max-depth")
    if args.target_error <= 0:
        raise ValueError("--target-error must be positive")
    if args.maxiters < 1:
        raise ValueError("--maxiters must be positive")


def main() -> int:
    args = parser().parse_args()
    validate(args)

    dmft.BACKEND = args.backend

    print("=" * 72)
    print("QCAOB-AIM KESTREL RUN")
    print("=" * 72)
    print(f"repo              : {REPO_ROOT}")
    print(f"host              : {socket.gethostname()}")
    print(f"python            : {sys.version.split()[0]}")
    print(f"platform          : {platform.platform()}")
    print(f"slurm_job_id      : {os.environ.get('SLURM_JOB_ID', 'not set')}")
    print(f"slurm_cpus/task   : {os.environ.get('SLURM_CPUS_PER_TASK', 'not set')}")
    print(f"backend           : {args.backend}")
    print(f"system_size       : {args.system_size}")
    print(f"seed              : {args.seed}")
    print(f"target_error      : {args.target_error}")
    print(f"starting_depth    : {args.starting_depth}")
    print(f"max_depth         : {args.max_depth}")
    print(f"ground_state_only : {args.ground_state_only}")
    print("=" * 72)
    sys.stdout.flush()

    start = time.perf_counter()

    dmft.run_gs_error_experiment(
        system_size=args.system_size,
        seed=args.seed,
        target_err=args.target_error,
        maxiters=args.maxiters,
        gs_gtol=args.gs_gtol,
        gf_gtol=args.gf_gtol,
        pre_empt_layers=args.max_depth,
        starting_depth=args.starting_depth,
        gs=args.ground_state_only,
        display=args.display,
        plot=args.plot,
    )

    elapsed = time.perf_counter() - start
    print()
    print("=" * 72)
    print("RUN COMPLETED")
    print(f"backend         : {dmft.BACKEND}")
    print(f"elapsed_seconds : {elapsed:.3f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
