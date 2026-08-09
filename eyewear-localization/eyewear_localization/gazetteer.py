"""Brand normalization and conservative OCR-to-brand matching."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class GazetteerMatch:
    brand: str
    score: float
    method: str


def normalize_text(value: str) -> str:
    """Lowercase, strip diacritics, and retain normalized word boundaries."""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _compact(value: str) -> str:
    return normalize_text(value).replace(" ", "")


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


class Gazetteer:
    """Match OCR strings without inventing labels absent from the vocabulary."""

    def __init__(self, brands: list[str] | tuple[str, ...] | set[str]):
        cleaned = {}
        for brand in brands:
            normalized = normalize_text(str(brand))
            if normalized:
                cleaned[normalized] = normalized
        self._brands = tuple(sorted(cleaned.values()))
        self._compact = {brand: _compact(brand) for brand in self._brands}

    @property
    def brands(self) -> tuple[str, ...]:
        return self._brands

    def match(self, text: str) -> GazetteerMatch | None:
        normalized = normalize_text(text)
        compact = normalized.replace(" ", "")
        if not compact:
            return None

        for brand in self._brands:
            brand_compact = self._compact[brand]
            if compact == brand_compact:
                return GazetteerMatch(brand, 1.0, "exact")

        # Token containment handles strings such as "RAY BAN EYEWEAR" while
        # keeping the gazetteer in control of which labels may be emitted. It
        # requires complete tokens, so an OCR truncation such as "CARTIE"
        # proceeds to the permitted edit-distance rule instead.
        text_tokens = set(normalized.split())
        for brand in self._brands:
            brand_tokens = set(normalize_text(brand).split())
            if brand_tokens and brand_tokens.issubset(text_tokens):
                return GazetteerMatch(brand, 0.94, "token_containment")

        # OCR often drops an ampersand/conjunction from compound labels (for
        # example ``DOLCE GABBANA``).  Permit that exact closed-set variant,
        # but never allow a single token such as ``KORS`` to stand in for
        # ``MICHAEL KORS``.
        for brand in self._brands:
            brand_tokens = set(normalize_text(brand).split())
            if "and" not in brand_tokens or len(brand_tokens) < 3:
                continue
            without_connector = brand_tokens - {"and"}
            if len(without_connector) >= 2 and without_connector.issubset(text_tokens):
                return GazetteerMatch(brand, 0.90, "token_containment_without_connector")

        # The specification permits edit distance <= 1.  Avoid accepting a
        # one-character OCR fragment as a brand by requiring length >= 4.
        best: GazetteerMatch | None = None
        for brand in self._brands:
            brand_compact = self._compact[brand]
            if min(len(compact), len(brand_compact)) < 4:
                continue
            distance = _edit_distance(compact, brand_compact)
            if distance <= 1:
                score = 1.0 - distance / max(len(compact), len(brand_compact))
                candidate = GazetteerMatch(brand, score, "edit_distance")
                if best is None or candidate.score > best.score:
                    best = candidate
        return best
