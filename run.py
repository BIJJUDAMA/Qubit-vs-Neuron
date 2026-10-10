#!/usr/bin/env python3
"""Unified pipeline runner for the QNN vs MLP case study.

Orchestrates:
1. Environment verification & dependencies.
2. Automated test suite execution.
3. PennyLane & PyTorch model benchmarks across all datasets.
4. Exporting summary metrics & LaTeX macros.
5. Automatic download of Tectonic compiler if missing.
6. Compilation of case_study.pdf.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = ROOT_DIR / "tools"


def ensure_tectonic() -> Path:
    """Ensure Tectonic executable is available; download if missing."""
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    system = platform.system().lower()
    machine = platform.machine().lower()
    exe_name = "tectonic.exe" if system == "windows" else "tectonic"
    tectonic_path = TOOLS_DIR / exe_name

    # Check local tools dir first
    if tectonic_path.is_file() and os.access(tectonic_path, os.X_OK if system != "windows" else os.R_OK):
        print(f"[pipeline] Found Tectonic at: {tectonic_path}")
        return tectonic_path

    # Check system PATH
    sys_tectonic = shutil.which("tectonic")
    if sys_tectonic:
        print(f"[pipeline] Found system Tectonic at: {sys_tectonic}")
        return Path(sys_tectonic)

    # Download pre-built release from GitHub
    print(f"[pipeline] Tectonic not found. Downloading for {system} ({machine})...")
    version = "0.15.0"
    if system == "windows":
        asset = f"tectonic-{version}-x86_64-pc-windows-msvc.zip"
    elif system == "darwin":
        asset = f"tectonic-{version}-aarch64-apple-darwin.tar.gz" if "arm" in machine else f"tectonic-{version}-x86_64-apple-darwin.tar.gz"
    else:
        asset = f"tectonic-{version}-x86_64-unknown-linux-musl.tar.gz"

    url = f"https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic@{version}/{asset}"
    archive_path = TOOLS_DIR / asset
    print(f"[pipeline] Fetching: {url}")
    urllib.request.urlretrieve(url, archive_path)

    if asset.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extract(exe_name, TOOLS_DIR)
    else:
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extract(exe_name, TOOLS_DIR)
        tectonic_path.chmod(0o755)

    if archive_path.is_file():
        archive_path.unlink()

    print(f"[pipeline] Successfully downloaded Tectonic to: {tectonic_path}")
    return tectonic_path


def run_pipeline(backend: str = "pennylane", seeds: int = 10, skip_pdf: bool = False) -> None:
    print("=" * 70)
    print("STEP 1: Running Automated Pytest Suite")
    print("=" * 70)
    test_cmd = [sys.executable, "-m", "pytest", "-q", "tests"]
    subprocess.run(test_cmd, cwd=str(ROOT_DIR), check=True)

    print("\n" + "=" * 70)
    print(f"STEP 2: Executing Model Benchmarks (backend={backend}, seeds={seeds})")
    print("=" * 70)
    main_cmd = [
        sys.executable,
        "main.py",
        "--backend",
        backend,
        "--seeds",
        str(seeds),
        "--qnn-epochs",
        "100",
        "--mlp-epochs",
        "100",
        "--datasets",
        "iris",
        "two_moons",
        "concentric_circles",
        "qdataset_hamiltonian",
        "--output-dir",
        "results",
    ]
    subprocess.run(main_cmd, cwd=str(ROOT_DIR), check=True)

    if skip_pdf:
        print("\n[pipeline] Skipping PDF compilation as requested.")
        return

    print("\n" + "=" * 70)
    print("STEP 3: Compiling LaTeX Manuscript into PDFs")
    print("=" * 70)
    tectonic_bin = ensure_tectonic()

    # Compile case_study.tex
    print("[pipeline] Building case_study.pdf...")
    subprocess.run([str(tectonic_bin), "case_study.tex"], cwd=str(ROOT_DIR), check=True)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Generated manuscript: {ROOT_DIR / 'case_study.pdf'}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="pennylane", choices=["pennylane", "torch"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--skip-pdf", action="store_true", help="Skip LaTeX compilation")
    args = parser.parse_args()
    run_pipeline(backend=args.backend, seeds=args.seeds, skip_pdf=args.skip_pdf)


if __name__ == "__main__":
    main()
