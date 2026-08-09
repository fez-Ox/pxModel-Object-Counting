#!/usr/bin/env python3
"""Automated Kaggle Kernel Execution & Output Fetcher.

Pushes eyewear_localization_kaggle.ipynb to Kaggle headlessly,
triggers GPU execution, polls for completion, and downloads all
generated visualization images & prediction JSON files directly to ./output/.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_ROOT / "kernel-metadata.json"
NOTEBOOK_FILE = PROJECT_ROOT / "eyewear_localization_kaggle.ipynb"
OUTPUT_DIR = PROJECT_ROOT / "output" / "kaggle_results"


def ensure_kaggle_installed() -> None:
    """Ensure kaggle CLI package is installed in the uv environment."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("Installing kaggle CLI via uv...")
        subprocess.check_call(["uv", "pip", "install", "kaggle"])


def check_credentials() -> bool:
    """Verify ~/.kaggle/kaggle.json exists."""
    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")

    if cred_path.exists() or (env_user and env_key):
        return True

    print("\n" + "=" * 65)
    print("ERROR: Kaggle credentials not found!")
    print("To enable headless 1-click execution on Kaggle:")
    print("  1. Go to https://www.kaggle.com/settings")
    print("  2. Click 'Create New Token' under API section")
    print("  3. Save the downloaded kaggle.json file to ~/.kaggle/kaggle.json")
    print("     or set environment variables KAGGLE_USERNAME and KAGGLE_KEY.")
    print("=" * 65 + "\n")
    return False


def get_or_create_metadata() -> str:
    """Read or create kernel-metadata.json."""
    default_username = os.environ.get("KAGGLE_USERNAME", "")
    if not default_username:
        cred_path = Path.home() / ".kaggle" / "kaggle.json"
        if cred_path.exists():
            try:
                data = json.loads(cred_path.read_text())
                default_username = data.get("username", "")
            except Exception:
                pass

    if not default_username:
        default_username = "your-kaggle-username"

    slug = "eyewear-localization-kaggle"
    kernel_id = f"{default_username}/{slug}"

    if METADATA_FILE.exists():
        try:
            meta = json.loads(METADATA_FILE.read_text())
            kernel_id = meta.get("id", kernel_id)
        except Exception:
            pass

    metadata_content = {
        "id": kernel_id,
        "title": "Eyewear Localization & Brand Attribution",
        "code_file": "eyewear_localization_kaggle.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
    }

    METADATA_FILE.write_text(json.dumps(metadata_content, indent=2))
    return kernel_id


def main() -> None:
    print("=== Automated Kaggle Remote Execution ===")
    ensure_kaggle_installed()

    if not check_credentials():
        sys.exit(1)

    kernel_id = get_or_create_metadata()
    print(f"Kernel Target: {kernel_id}")
    print("Pushing notebook to Kaggle for GPU execution...")

    # Push kernel
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(PROJECT_ROOT)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Push failed:\n{result.stderr}")
        if "your-kaggle-username" in kernel_id:
            print("\nPlease update 'id' in kernel-metadata.json with your actual Kaggle username.")
        sys.exit(1)

    print(f"Success! {result.stdout.strip()}")
    print("Waiting for cloud GPU execution to complete...")

    # Poll status
    start_time = time.time()
    last_status = ""
    while True:
        status_proc = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "status", kernel_id],
            capture_output=True,
            text=True,
        )
        status_line = status_proc.stdout.strip()
        if status_line != last_status:
            print(f"[{int(time.time() - start_time)}s] Status: {status_line}")
            last_status = status_line

        status_lower = status_line.lower()
        if "complete" in status_lower:
            print("\nExecution finished successfully!")
            break
        if "error" in status_lower or "failed" in status_lower:
            print(f"\nExecution failed on Kaggle: {status_line}")
            sys.exit(1)

        time.sleep(15)

    # Fetch results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading output results to {OUTPUT_DIR}...")

    out_proc = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "output", kernel_id, "-p", str(OUTPUT_DIR)],
        capture_output=True,
        text=True,
    )
    if out_proc.returncode == 0:
        print(f"\nResults downloaded successfully to {OUTPUT_DIR}/!")
        files = list(OUTPUT_DIR.glob("*"))
        for f in files:
            print(f"  - {f.name}")
    else:
        print(f"Download failed: {out_proc.stderr}")


if __name__ == "__main__":
    main()
