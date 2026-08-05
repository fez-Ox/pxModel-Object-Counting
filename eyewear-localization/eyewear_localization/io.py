"""Input expansion and JSON output helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import mimetypes
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator
from urllib.parse import unquote, urlparse
import urllib.error
import urllib.request

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageSource:
    source: str
    path: Path


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def expand_inputs(items: list[str], *, recursive: bool = False) -> list[str]:
    """Expand local folders while preserving URL and file input order."""
    result: list[str] = []
    for item in items:
        if is_url(item):
            result.append(item)
            continue
        path = Path(item).expanduser()
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            result.extend(str(candidate) for candidate in sorted(iterator) if is_image_file(candidate))
        else:
            if not path.exists():
                raise FileNotFoundError(path)
            result.append(str(path))
    return result


def _download(url: str, directory: Path, timeout: int) -> Path:
    parsed = urlparse(url)
    suffix = Path(unquote(parsed.path)).suffix
    if not suffix:
        content_type = mimetypes.guess_type(url)[0] or "image/jpeg"
        suffix = mimetypes.guess_extension(content_type) or ".jpg"
    descriptor, name = tempfile.mkstemp(prefix="localization_", suffix=suffix, dir=directory)
    os.close(descriptor)
    destination = Path(name)
    request = urllib.request.Request(url, headers={"User-Agent": "eyewear-localization/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except (urllib.error.URLError, TimeoutError, OSError):
        destination.unlink(missing_ok=True)
        raise
    return destination


@contextmanager
def materialize_inputs(
    items: list[str], *, recursive: bool = False, timeout: int = 30
) -> Iterator[list[ImageSource]]:
    """Materialize local paths and URLs and clean URL downloads afterwards."""
    expanded = expand_inputs(items, recursive=recursive)
    with tempfile.TemporaryDirectory(prefix="eyewear_localization_") as temporary:
        temporary_path = Path(temporary)
        sources: list[ImageSource] = []
        for item in expanded:
            if is_url(item):
                sources.append(ImageSource(item, _download(item, temporary_path, timeout)))
            else:
                sources.append(ImageSource(item, Path(item).expanduser().resolve()))
        yield sources


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
