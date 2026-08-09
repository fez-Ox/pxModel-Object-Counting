#!/usr/bin/env python3
"""Download the native SAM3 checkpoint directly for notebook use."""

from __future__ import annotations

import argparse
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt"
APP_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = APP_ROOT / "checkpoints" / "sam3.pt"


def _get_token(explicit_token: str | None = None) -> str | None:
    if explicit_token:
        return explicit_token
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token

    # Kaggle notebooks can expose a secret without requiring `hf auth login`.
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def download_model(
    *,
    url: str,
    output: Path,
    force: bool,
    timeout: int,
    token: str | None = None,
) -> Path:
    output = output.expanduser().resolve()
    if output.exists() and not force:
        print(f"SAM3 checkpoint already exists: {output}")
        print("Use --force to download it again.")
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    token = _get_token(token)
    if not token:
        raise SystemExit(
            "No Hugging Face token found. Add an approved HF_TOKEN Kaggle Secret "
            "or pass --token explicitly."
        )

    try:
        from huggingface_hub import hf_hub_download

        print("Downloading SAM3 checkpoint via hf_hub_download...")
        downloaded_path = hf_hub_download(
            repo_id="facebook/sam3",
            filename="sam3.pt",
            token=token,
            local_dir=str(output.parent),
            force_download=force,
        )
        downloaded = Path(downloaded_path)
        if downloaded != output and downloaded.exists():
            shutil.move(str(downloaded), str(output))
        print(f"Done: {output}")
        return output
    except Exception as exc:
        # Keep a direct fallback for Kaggle images with an older or partially
        # broken huggingface_hub. The fallback handles the gated URL's 302
        # redirect explicitly: the bearer token is sent to Hugging Face, while
        # the signed CDN URL is fetched without forwarding that header.
        print(f"hf_hub_download failed ({type(exc).__name__}), trying direct HTTP...")

    temporary = output.with_name(output.name + ".part")
    temporary.unlink(missing_ok=True)
    headers = {"User-Agent": "sam3-verbose-counting/1.0", "Authorization": f"Bearer {token}"}
    request = urllib.request.Request(url, headers=headers)

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    print(f"Downloading: {url}")
    print(f"Destination: {output}")
    try:
        try:
            response = opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code not in {301, 302, 303, 307, 308}:
                raise
            location = exc.headers.get("Location")
            if not location:
                raise
            # The redirect URL is already signed by Hugging Face.
            response = urllib.request.urlopen(
                urllib.request.Request(location, headers={"User-Agent": headers["User-Agent"]}),
                timeout=timeout,
            )
        with response, temporary.open("wb") as stream:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded_bytes = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                downloaded_bytes += len(chunk)
                if total:
                    percent = downloaded_bytes * 100 / total
                    print(
                        f"\r{downloaded_bytes / (1024 ** 2):.1f} / "
                        f"{total / (1024 ** 2):.1f} MiB ({percent:.1f}%)",
                        end="",
                    )
                else:
                    print(f"\r{downloaded_bytes / (1024 ** 2):.1f} MiB", end="")
            print()
        shutil.move(str(temporary), str(output))
    except urllib.error.HTTPError as exc:
        temporary.unlink(missing_ok=True)
        if exc.code == 401:
            raise SystemExit(
                "SAM3 returned HTTP 401. The official SAM3 checkpoint is gated. "
                "Request/receive access on Hugging Face, then add the approved "
                "token as a Kaggle Secret named HF_TOKEN (or pass --token)."
            ) from exc
        raise SystemExit(f"SAM3 download failed: {exc}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"SAM3 download failed: {exc}") from exc

    if output.stat().st_size == 0:
        output.unlink(missing_ok=True)
        raise SystemExit("SAM3 download failed: checkpoint file is empty")

    print(f"Done: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the native SAM3 checkpoint directly over HTTP."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Direct checkpoint URL")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="Local checkpoint path")
    parser.add_argument("--timeout", type=int, default=120, help="Download timeout in seconds")
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token; Kaggle Secret HF_TOKEN is auto-detected",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing checkpoint")
    args = parser.parse_args()

    download_model(
        url=args.url,
        output=Path(args.output),
        force=args.force,
        timeout=args.timeout,
        token=args.token,
    )


if __name__ == "__main__":
    main()
