#!/usr/bin/env python3
"""Download the CountAnything checkpoint to the default inference path.

Usage:
  python download_checkpoint.py
  python download_checkpoint.py --force
"""

import argparse
import shutil
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_URL = "https://huggingface.co/MengqiLei/count-anything/resolve/main/count_anything.pt"
DEFAULT_OUTPUT = REPO_ROOT / "checkpoints" / "count_anything.pt"


def _download(url: str, output: Path, force: bool = False) -> None:
    output = output.expanduser().resolve()
    if output.exists() and not force:
        print(f"Checkpoint already exists: {output}")
        print("Use --force to re-download.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(output.suffix + ".part")
    tmp_path.unlink(missing_ok=True)

    request = urllib.request.Request(url, headers={"User-Agent": "count-anything-downloader/1.0"})
    print(f"Downloading: {url}")
    print(f"Destination: {output}")

    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(tmp_path, "wb") as f:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100 / total
                    print(f"\r{downloaded / (1024 ** 2):.1f} / {total / (1024 ** 2):.1f} MiB ({percent:.1f}%)", end="")
                else:
                    print(f"\r{downloaded / (1024 ** 2):.1f} MiB", end="")
            print()
        shutil.move(str(tmp_path), str(output))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp_path.unlink(missing_ok=True)
        raise SystemExit(f"Download failed: {exc}") from exc

    if output.stat().st_size == 0:
        output.unlink(missing_ok=True)
        raise SystemExit("Download failed: checkpoint file is empty")

    print(f"Done: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download count_anything.pt for CountAnything inference.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Checkpoint URL")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="Output checkpoint path")
    parser.add_argument("--force", action="store_true", help="Re-download even if the checkpoint already exists")
    args = parser.parse_args()

    _download(args.url, Path(args.output), args.force)


if __name__ == "__main__":
    main()
