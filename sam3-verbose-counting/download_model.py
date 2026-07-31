#!/usr/bin/env python3
"""Download the native SAM3 checkpoint into the standalone application.

The SAM3 repository may require a Hugging Face account, accepted model terms,
and an access token. Authenticate first with ``hf auth login`` or set
``HF_TOKEN`` in the notebook environment.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

DEFAULT_REPO_ID = "facebook/sam3"
DEFAULT_FILENAME = "sam3.pt"
APP_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = APP_ROOT / "checkpoints" / DEFAULT_FILENAME


def download_model(
    *,
    repo_id: str,
    filename: str,
    output: Path,
    revision: str,
    force: bool,
) -> Path:
    output = output.expanduser().resolve()
    if output.exists() and not force:
        print(f"SAM3 checkpoint already exists: {output}")
        print("Use --force to download it again.")
        return output

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on environment setup
        raise SystemExit("Install dependencies first with: uv sync") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    temporary.unlink(missing_ok=True)

    token = os.environ.get("HF_TOKEN")
    print(f"Downloading {repo_id}/{filename} ({revision=})")
    print(f"Destination: {output}")
    try:
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=token,
        )
        shutil.copyfile(cached, temporary)
        temporary.replace(output)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            "SAM3 download failed. Verify Hugging Face access, accepted model "
            f"terms, and HF_TOKEN/authentication: {exc}"
        ) from exc

    if output.stat().st_size == 0:
        output.unlink(missing_ok=True)
        raise SystemExit("SAM3 download failed: checkpoint file is empty")

    print(f"Done: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the native SAM3 checkpoint for verbose-prompt counting."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repository")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="Repository filename")
    parser.add_argument("--revision", default="main", help="Hugging Face revision")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="Local checkpoint path")
    parser.add_argument("--force", action="store_true", help="Replace an existing checkpoint")
    args = parser.parse_args()

    download_model(
        repo_id=args.repo_id,
        filename=args.filename,
        output=Path(args.output),
        revision=args.revision,
        force=args.force,
    )


if __name__ == "__main__":
    main()
