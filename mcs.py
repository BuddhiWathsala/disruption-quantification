from __future__ import annotations

import os
import subprocess
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # main.py runs its pipeline on import, so only import for type hints
    from main import Company


def write_dimacs(company: Company):
    """
    Write graph g to a temporary DIMACS-format file (1-indexed vertices,
    labels on 'n' lines) and return the file path.
    """
    adjlist = company.adjmat  # adjacency list: adjlist[i] holds i's neighbours
    n = len(adjlist)
    # directed: j in adjlist[i] is the arc i -> j (duplicate entries dropped)
    edges = sorted({(i, j) for i in range(n) for j in adjlist[i]})

    with tempfile.NamedTemporaryFile("w", prefix="company_", suffix=".dimacs",
                                     delete=False, encoding="utf-8") as f:
        f.write(f"p edge {n} {len(edges)}\n")
        for i, label in enumerate(company.labels):
            f.write(f"n {i + 1} {label}\n")
        for i, j in edges:
            f.write(f"e {i + 1} {j + 1}\n")
    return f.name


def run_symsplit(pattern, target, exe=None,
                 timeout_s=100, u=0, w=0, labelled=True,
                 method="min_max", run_timeout=None):
    """
    Run symsplit on two DIMACS files and return (returncode, stdout, stderr).

    Mirrors:
      symsplit min_max <pattern> <target> -l -t 100 -u 0 -w 0 -d -p 0
    """
    if exe is None:
        exe = os.environ.get("FMCIS_SYMSPLIT_BIN", "./symsplit")

    # -l selects LAD input format (not "labelled"), which can't parse our DIMACS files
    cmd = [exe, method, pattern, target, "-t", str(timeout_s), "-d", "-p", str(0)]

    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
        timeout=run_timeout
    )
    return result.returncode, result.stdout, result.stderr
