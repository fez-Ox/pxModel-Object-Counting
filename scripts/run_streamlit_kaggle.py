#!/usr/bin/env python3
"""Validate, push, execute, and stream logs for the Streamlit Kaggle GPU Web UI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_ROOT / "kernel-metadata-streamlit.json"
NOTEBOOK_FILE = PROJECT_ROOT / "eyewear_localization_streamlit_kaggle.ipynb"
OUTPUT_DIR = PROJECT_ROOT / "output" / "kaggle_streamlit_results"
DEFAULT_KERNEL_ID = "faizankhan101/eyewear-localization-streamlit-dashboard"


def _run(command: Sequence[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=check)


def _kaggle_command() -> list[str]:
    try:
        import kaggle  # noqa: F401
        return [sys.executable, "-m", "kaggle"]
    except ImportError:
        binary = shutil.which("kaggle")
        if binary:
            return [binary]
    raise RuntimeError("Kaggle CLI is unavailable. Install with `uv pip install kaggle`.")


def check_credentials() -> None:
    if os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        return
    credential_path = Path.home() / ".kaggle" / "kaggle.json"
    if credential_path.exists():
        try:
            data = json.loads(credential_path.read_text(encoding="utf-8"))
            if data.get("username") and data.get("key"):
                return
        except (OSError, ValueError):
            pass
    raise RuntimeError("Kaggle credentials unavailable. Check ~/.kaggle/kaggle.json")


def _credential_username() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    credential_path = Path.home() / ".kaggle" / "kaggle.json"
    if credential_path.exists():
        try:
            value = json.loads(credential_path.read_text(encoding="utf-8"))
            if value.get("username"):
                return str(value["username"])
        except (OSError, ValueError):
            pass
    return "faizankhan101"


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token.strip()
    path = Path.home() / ".kaggle" / "hf_token"
    if path.exists():
        val = path.read_text(encoding="utf-8").strip()
        if val:
            return val
    raise RuntimeError("No HF_TOKEN found.")


def set_notebook_hf_token(notebook_path: Path, token_value: str) -> None:
    """Inject a token into a temporary notebook copy, never the working tree."""
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    replacement = json.dumps(token_value)
    replaced = 0
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        updated, count = re.subn(
            r"(?m)^(\s*HF_TOKEN_FALLBACK\s*=\s*).*$",
            lambda match: match.group(1) + replacement,
            source,
        )
        if count:
            replaced += count
            cell["source"] = updated.splitlines(keepends=True)
    if replaced == 0:
        raise RuntimeError("Could not find HF_TOKEN_FALLBACK in notebook.")
    notebook_path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def get_or_create_metadata() -> str:
    username = _credential_username()
    kernel_id = f"{username}/eyewear-localization-streamlit-dashboard"
    metadata_content = {
        "id": kernel_id,
        "title": "Eyewear Localization Streamlit Dashboard",
        "code_file": NOTEBOOK_FILE.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": ["faizankhan101/eyewear-test-samples"],
        "kernel_sources": [],
        "competition_sources": [],
    }
    METADATA_FILE.write_text(json.dumps(metadata_content, indent=2) + "\n", encoding="utf-8")
    return kernel_id


def ensure_remote_head() -> None:
    local = _run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], check=True).stdout.strip()
    remote_proc = _run(["git", "-C", str(PROJECT_ROOT), "ls-remote", "origin", "refs/heads/main"])
    if remote_proc.returncode != 0:
        raise RuntimeError(f"Could not verify origin/main: {remote_proc.stderr.strip()}")
    remote = remote_proc.stdout.split()[0] if remote_proc.stdout.split() else ""
    if not remote or remote != local:
        raise RuntimeError(
            f"origin/main ({remote or 'unknown'}) does not match local HEAD ({local}). "
            "Commit and push before running Kaggle."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-timeout", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=10)
    args = parser.parse_args(argv)

    print("=== Automated Kaggle Remote Execution (Streamlit App) ===")
    ensure_remote_head()
    check_credentials()
    kaggle = _kaggle_command()
    kernel_id = get_or_create_metadata()
    token = get_hf_token()

    # Create temporary push folder
    temporary = tempfile.TemporaryDirectory(prefix="streamlit-kaggle-push-")
    directory = Path(temporary.name)
    shutil.copy2(NOTEBOOK_FILE, directory / NOTEBOOK_FILE.name)
    # Inject the approved HF token (SAM3 is a gated checkpoint) into the
    # temporary notebook copy only; the working tree never holds it.
    set_notebook_hf_token(directory / NOTEBOOK_FILE.name, token)
    
    # Patch metadata to use current folder filename
    meta = json.loads(METADATA_FILE.read_text())
    (directory / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    try:
        print("Pushing Streamlit notebook to Kaggle GPU...")
        pushed = _run([*kaggle, "kernels", "push", "-p", temporary.name])
        if pushed.returncode != 0:
            raise RuntimeError(f"Kaggle push failed:\n{pushed.stdout}\n{pushed.stderr}")
        print(pushed.stdout.strip())
    finally:
        temporary.cleanup()

    print("Waiting for Streamlit server setup and URL generation...")
    started = time.monotonic()
    url_printed = False

    while time.monotonic() - started < args.wait_timeout:
        # Check logs for localtunnel URL
        log_proc = _run([*kaggle, "kernels", "logs", kernel_id])
        log_text = log_proc.stdout + log_proc.stderr

        if "STREAMLIT PUBLIC WEB APP URL" in log_text or "your url is:" in log_text.lower():
            print("\n==========================================================================================")
            print("🚀 STREAMLIT APP LIVE ON KAGGLE GPU!")
            for line in log_text.splitlines():
                if "STREAMLIT PUBLIC WEB APP URL" in line or "your url is:" in line.lower():
                    print(line)
            print("==========================================================================================\n")
            url_printed = True

        status_proc = _run([*kaggle, "kernels", "status", kernel_id])
        status = status_proc.stdout.strip()
        print(f"[{int(time.monotonic() - started)}s] Status: {status}", flush=True)

        if "complete" in status.lower() or "error" in status.lower() or "failed" in status.lower():
            break

        time.sleep(args.poll_seconds)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, TimeoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
