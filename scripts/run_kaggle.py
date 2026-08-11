#!/usr/bin/env python3
"""Validate, push, execute, and download the eyewear Kaggle notebook.

The notebook clones the repository at ``origin/main``. This runner therefore
requires the local HEAD to already be pushed, validates the notebook before
uploading it, and uploads a temporary copy so the local notebook is never
modified with a secret.

Credentials are read from the normal Kaggle configuration and from
``~/.kaggle/hf_token``. Secrets are never printed or committed.
"""

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
METADATA_FILE = PROJECT_ROOT / "kernel-metadata.json"
NOTEBOOK_FILE = PROJECT_ROOT / "eyewear_localization_kaggle.ipynb"
OUTPUT_DIR = PROJECT_ROOT / "output" / "kaggle_results"
DEFAULT_KERNEL_ID = "faizankhan101/eyewear-localization-brand-attribution"


def _run(command: Sequence[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=check)


def _kaggle_command() -> list[str]:
    """Return a working Kaggle CLI command for the current environment."""
    try:
        import kaggle  # noqa: F401

        return [sys.executable, "-m", "kaggle"]
    except ImportError:
        binary = shutil.which("kaggle")
        if binary:
            return [binary]
    raise RuntimeError(
        "Kaggle CLI is unavailable. Run this script with the project UV environment "
        "or install it using `uv pip install kaggle`."
    )


def ensure_kaggle_installed() -> list[str]:
    """Use an existing CLI, or install it into the active UV environment."""
    try:
        return _kaggle_command()
    except RuntimeError:
        uv = shutil.which("uv")
        if not uv:
            raise
        subprocess.run([uv, "pip", "install", "--python", sys.executable, "kaggle"], check=True)
        return _kaggle_command()


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
    if METADATA_FILE.exists():
        try:
            kernel_id = json.loads(METADATA_FILE.read_text(encoding="utf-8")).get("id", "")
            if "/" in kernel_id:
                return kernel_id.split("/", 1)[0]
        except (OSError, ValueError):
            pass
    return ""


def check_credentials() -> None:
    """Check that Kaggle credentials have usable values, not just a file."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    credential_path = Path.home() / ".kaggle" / "kaggle.json"
    if credential_path.exists():
        try:
            data = json.loads(credential_path.read_text(encoding="utf-8"))
            if data.get("username") and data.get("key"):
                return
        except (OSError, ValueError):
            pass
    access_token = Path.home() / ".kaggle" / "access_token"
    if access_token.exists() and access_token.read_text(encoding="utf-8").strip() and _credential_username():
        # The legacy CLI accepts username/key; convert the new token once so
        # the same credential works across Kaggle CLI versions.
        credential_path.parent.mkdir(parents=True, exist_ok=True)
        credential_path.write_text(
            json.dumps(
                {"username": _credential_username(), "key": access_token.read_text(encoding="utf-8").strip()},
                indent=2,
            ),
            encoding="utf-8",
        )
        credential_path.chmod(0o600)
        return
    raise RuntimeError(
        "Kaggle credentials are unavailable. Set KAGGLE_API_TOKEN or "
        "KAGGLE_USERNAME/KAGGLE_KEY, or place kaggle.json in ~/.kaggle/."
    )


def get_hf_token(token_file: Path | None = None) -> str:
    """Read the approved local HF token without logging its value."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token.strip()
    path = token_file or (Path.home() / ".kaggle" / "hf_token")
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    raise RuntimeError(
        f"No Hugging Face token found. Add an approved token to {path} or set HF_TOKEN."
    )


def get_or_create_metadata() -> str:
    """Write metadata with the configured Kaggle owner and return kernel id."""
    username = _credential_username()
    if not username:
        raise RuntimeError("Could not determine Kaggle username for kernel metadata.")
    kernel_id = f"{username}/eyewear-localization-brand-attribution"
    if METADATA_FILE.exists():
        try:
            existing = json.loads(METADATA_FILE.read_text(encoding="utf-8")).get("id", "")
            if existing and existing.split("/", 1)[0] == username:
                kernel_id = existing
        except (OSError, ValueError):
            pass
    metadata_content = {
        "id": kernel_id,
        "title": "Eyewear Localization Brand Attribution",
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


def validate_notebook(path: Path = NOTEBOOK_FILE) -> None:
    """Perform the same cheap checks locally that would otherwise fail remotely."""
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read notebook {path}: {exc}") from exc
    if notebook.get("nbformat", 0) < 4:
        raise RuntimeError("Notebook is not a valid nbformat v4 notebook.")
    code = ["".join(cell.get("source", [])) for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if not any("SAM3_CHECKPOINT" in source for source in code):
        raise RuntimeError("Notebook is missing the SAM3 setup/inference cells.")
    if not any("HF_TOKEN_FALLBACK" in source for source in code):
        raise RuntimeError("Notebook has no HF_TOKEN_FALLBACK hook for headless execution.")
    for index, source in enumerate(code):
        try:
            compile(source, f"{path}:cell-{index}", "exec")
        except SyntaxError as exc:
            raise RuntimeError(f"Notebook code cell {index} has a syntax error: {exc}") from exc


def ensure_remote_head() -> None:
    """The notebook clones origin/main, so prevent stale-code execution."""
    local = _run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], check=True).stdout.strip()
    remote_proc = _run(["git", "-C", str(PROJECT_ROOT), "ls-remote", "origin", "refs/heads/main"])
    if remote_proc.returncode != 0:
        raise RuntimeError(f"Could not verify origin/main: {remote_proc.stderr.strip()}")
    remote = remote_proc.stdout.split()[0] if remote_proc.stdout.split() else ""
    if not remote or remote != local:
        raise RuntimeError(
            f"origin/main ({remote or 'unknown'}) does not match local HEAD ({local}). "
            "Commit and push the code before running Kaggle."
        )


def set_notebook_runtime_options(
    notebook_path: Path,
    *,
    target_image: str | None = None,
    max_images: int | None = None,
) -> None:
    """Patch bounded image selection in the temporary notebook copy."""
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    replacements = 0
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        target_val = json.dumps(target_image) if target_image is not None else "None"
        source, count1 = re.subn(
            r"(?m)^TARGET_IMAGE_NAME\s*=\s*.*$",
            f"TARGET_IMAGE_NAME = {target_val}",
            source,
        )
        replacements += count1
        max_val = f"{int(max_images)}" if max_images is not None else "None"
        source, count2 = re.subn(
            r"(?m)^MAX_IMAGES\s*=\s*.*$",
            f"MAX_IMAGES = {max_val}",
            source,
        )
        replacements += count2
        cell["source"] = source.splitlines(keepends=True)
    if replacements < 2:
        raise RuntimeError("Could not patch the requested notebook runtime options.")
    notebook_path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


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


def _push_directory(
    token: str,
    *,
    target_image: str | None = None,
    max_images: int | None = None,
) -> tempfile.TemporaryDirectory[str]:
    """Create a minimal push directory and inject temporary options/token."""
    temporary = tempfile.TemporaryDirectory(prefix="eyewear-kaggle-push-")
    directory = Path(temporary.name)
    shutil.copy2(NOTEBOOK_FILE, directory / NOTEBOOK_FILE.name)
    shutil.copy2(METADATA_FILE, directory / METADATA_FILE.name)
    set_notebook_runtime_options(
        directory / NOTEBOOK_FILE.name,
        target_image=target_image,
        max_images=max_images,
    )
    set_notebook_hf_token(directory / NOTEBOOK_FILE.name, token)
    if token not in (directory / NOTEBOOK_FILE.name).read_text(encoding="utf-8"):
        raise RuntimeError("HF token injection verification failed.")
    return temporary


def _status(kaggle: list[str], kernel_id: str) -> tuple[int, str]:
    for attempt in range(3):
        process = _run([*kaggle, "kernels", "status", kernel_id])
        if process.returncode == 0:
            return process.returncode, (process.stdout + process.stderr).strip()
        time.sleep(3)
    return process.returncode, (process.stdout + process.stderr).strip()


def _delete_kernel(kaggle: list[str], kernel_id: str) -> str:
    """Cancel and delete the remote worker so a timeout cannot burn quota."""
    process = _run([*kaggle, "kernels", "delete", "-y", kernel_id])
    message = (process.stdout + process.stderr).strip()
    if process.returncode != 0:
        return f"delete failed: {message}"
    return message or "delete requested"


def _save_remote_logs(kaggle: list[str], kernel_id: str, output_dir: Path) -> Path | None:
    """Persist logs before deleting an errored/timed-out kernel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    process = _run([*kaggle, "kernels", "logs", kernel_id])
    payload = process.stdout + process.stderr
    if not payload.strip():
        return None
    path = output_dir / f"{kernel_id.rsplit('/', 1)[-1]}-remote.log"
    path.write_text(payload, encoding="utf-8")
    return path

def _save_remote_output(kaggle: list[str], kernel_id: str, output_dir: Path) -> bool:
    """Download partial artifacts before deleting a failed worker."""
    output_dir.mkdir(parents=True, exist_ok=True)
    process = _run([*kaggle, "kernels", "output", kernel_id, "-p", str(output_dir), "--force"])
    if process.returncode == 0:
        print(f"Saved partial remote output: {output_dir}", flush=True)
        return True
    message = (process.stdout + process.stderr).strip()
    if message:
        print(f"WARNING: partial remote output unavailable: {message[-1000:]}", flush=True)
    return False


def _cleanup_after_abort(
    kaggle: list[str],
    kernel_id: str,
    *,
    reason: str,
    grace_seconds: int,
    output_dir: Path,
) -> None:
    try:
        log_path = _save_remote_logs(kaggle, kernel_id, output_dir)
        if log_path:
            print(f"Saved remote logs: {log_path}", flush=True)
        _save_remote_output(kaggle, kernel_id, output_dir)
    except OSError as exc:
        print(f"WARNING: could not save remote logs: {exc}", flush=True)
    print(f"Remote cleanup ({reason}): {_delete_kernel(kaggle, kernel_id)}", flush=True)
    deadline = time.monotonic() + max(0, grace_seconds)
    last_status = ""
    while time.monotonic() < deadline:
        rc, status = _status(kaggle, kernel_id)
        if status != last_status:
            print(f"[cleanup] {status}", flush=True)
            last_status = status
        lowered = status.lower()
        # A deleted private kernel commonly returns permission/not-found here;
        # that is a successful quota cleanup, not a second failure.
        if rc != 0 or any(word in lowered for word in (
            "complete",
            "cancel_acknowledged",
            "cancelled",
            "error",
            "failed",
            "not found",
            "denied",
        )):
            return
        time.sleep(3)
    print("WARNING: remote cleanup grace period expired; verify kernel status manually.", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-id", default=None, help="Override the metadata kernel id")
    parser.add_argument("--hf-token-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--wait-timeout", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument(
        "--cleanup-grace-seconds",
        type=int,
        default=45,
        help="After timeout/cancel, wait this long after deleting the kernel",
    )
    parser.add_argument("--target-image", default=None, help="Temporary notebook target image filename")
    parser.add_argument("--max-images", type=int, default=None, help="Temporary notebook image-count limit")
    parser.add_argument("--skip-remote-head-check", action="store_true")
    args = parser.parse_args(argv)

    print("=== Automated Kaggle Remote Execution ===")
    validate_notebook()
    if not args.skip_remote_head_check:
        ensure_remote_head()
    check_credentials()
    kaggle = ensure_kaggle_installed()
    kernel_id = args.kernel_id or get_or_create_metadata()
    token = get_hf_token(args.hf_token_file)

    # Validate the Kaggle API before uploading the notebook. The target may
    # legitimately be absent after a failed/stopped run; `kernels list` checks
    # credentials without requiring that particular private kernel to exist.
    rc, status = _status(kaggle, kernel_id)
    if rc != 0:
        credential_probe = _run([*kaggle, "kernels", "list", "--mine", "--page-size", "1"])
        if credential_probe.returncode != 0:
            raise RuntimeError(
                f"Kaggle API authentication failed: {status}\\n{credential_probe.stderr.strip()}"
            )
        print("Kaggle API: authenticated; target kernel will be created/updated")
    else:
        print(f"Kaggle API: authenticated; kernel={kernel_id}")
    print("HF token: available (value not printed)")

    temporary = _push_directory(
        token,
        target_image=args.target_image,
        max_images=args.max_images,
    )
    try:
        print("Pushing validated notebook to Kaggle...")
        pushed = _run([*kaggle, "kernels", "push", "-p", temporary.name])
        if pushed.returncode != 0:
            raise RuntimeError(f"Kaggle push failed:\n{pushed.stdout}\n{pushed.stderr}")
        print(pushed.stdout.strip())
    finally:
        temporary.cleanup()

    if args.wait_timeout <= 0:
        raise ValueError("--wait-timeout must be positive")
    print("Waiting for cloud GPU execution...")
    started = time.monotonic()
    last_status = ""
    try:
        while time.monotonic() - started < args.wait_timeout:
            rc, status = _status(kaggle, kernel_id)
            if status != last_status:
                print(f"[{int(time.monotonic() - started)}s] {status}", flush=True)
                last_status = status
            if rc != 0:
                raise RuntimeError(f"Kaggle status failed: {status}")
            lowered = status.lower()
            if "complete" in lowered:
                break
            if any(word in lowered for word in ("cancel", "stopped", "aborted", "error", "failed")):
                raise RuntimeError(f"Kaggle execution terminated: {status}")
            time.sleep(max(1, args.poll_seconds))
        else:
            raise TimeoutError(f"Kaggle kernel did not finish within {args.wait_timeout}s.")
    except (TimeoutError, KeyboardInterrupt) as exc:
        _cleanup_after_abort(
            kaggle,
            kernel_id,
            reason="timeout" if isinstance(exc, TimeoutError) else "interrupt",
            grace_seconds=args.cleanup_grace_seconds,
            output_dir=args.output,
        )
        raise
    except RuntimeError:
        _cleanup_after_abort(
            kaggle,
            kernel_id,
            reason="remote termination/error",
            grace_seconds=args.cleanup_grace_seconds,
            output_dir=args.output,
        )
        raise

    args.output.mkdir(parents=True, exist_ok=True)
    output = _run([*kaggle, "kernels", "output", kernel_id, "-p", str(args.output), "--force"])
    if output.returncode != 0:
        raise RuntimeError(f"Kaggle output download failed:\n{output.stdout}\n{output.stderr}")
    print(f"Results downloaded to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, TimeoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
