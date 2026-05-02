"""IPA string normalisation: strip suprasegmental markers and NFKD-normalise."""

import dataclasses
import re
import unicodedata

from phone_similarity._dispatch import (
    HAS_CYTHON_CLEAN,
    cy_clean_phones as _cy_clean_phones,
    cy_extract_stress_marks as _cy_extract_stress_marks,
)

# Marker character sets
_STRESS_CHARS = "ˈˌ"  # primary + secondary stress
_LENGTH_CHARS = "ːˑ"  # long + half-long
_LIAISON_CHARS = "‿"  # liaison / linking

# Pre-compiled patterns for each category
_PAT_STRESS = re.compile(f"[{re.escape(_STRESS_CHARS)}]")
_PAT_LENGTH = re.compile(f"[{re.escape(_LENGTH_CHARS)}]")
_PAT_LIAISON = re.compile(f"[{re.escape(_LIAISON_CHARS)}]")

# Legacy "strip everything" pattern
_STRIP_PATTERN = re.compile(r"[ˈˌːˑ‿]")


# CleanConfig
@dataclasses.dataclass(frozen=True)
class CleanConfig:
    """Per-language IPA normalisation settings (which suprasegmental categories to strip)."""

    strip_stress: bool = True
    strip_length: bool = True
    strip_liaison: bool = True
    nfkd: bool = True

    @property
    def strip_all(self) -> bool:
        """True when every suprasegmental category is being stripped."""
        return self.strip_stress and self.strip_length and self.strip_liaison


STRIP_ALL = CleanConfig()
"""Default config: strip everything (backward compatible)."""

PRESERVE_STRESS = CleanConfig(strip_stress=False)
"""Keep stress markers, strip length and liaison."""

PRESERVE_LENGTH = CleanConfig(strip_length=False)
"""Keep length markers, strip stress and liaison."""

PRESERVE_ALL = CleanConfig(
    strip_stress=False,
    strip_length=False,
    strip_liaison=False,
)
"""Keep all suprasegmental markers (only NFKD-normalise)."""


# Public API
def clean_phones(
    ipa: str,
    *,
    preserve_stress: bool = False,
    config: CleanConfig | None = None,
) -> str:
    """Remove suprasegmental markers and NFKD-normalise an IPA string."""
    if config is not None:
        cfg = config
    elif preserve_stress:
        cfg = PRESERVE_STRESS
    else:
        cfg = STRIP_ALL

    # Fast Cython path for the common strip-all case
    if HAS_CYTHON_CLEAN and cfg is STRIP_ALL:
        return _cy_clean_phones(ipa, True, True, True, True)

    if cfg.nfkd:
        ipa = unicodedata.normalize("NFKD", ipa)

    if cfg.strip_all:
        return _STRIP_PATTERN.sub("", ipa)

    if cfg.strip_stress:
        ipa = _PAT_STRESS.sub("", ipa)
    if cfg.strip_length:
        ipa = _PAT_LENGTH.sub("", ipa)
    if cfg.strip_liaison:
        ipa = _PAT_LIAISON.sub("", ipa)
    return ipa


# Stress extraction helpers
_PRIMARY = "ˈ"
_SECONDARY = "ˌ"


def extract_stress_marks(ipa: str) -> list[tuple[int, str]]:
    """Return ``[(clean_index, 'primary'|'secondary')]`` for stress markers in *ipa*."""
    if HAS_CYTHON_CLEAN:
        return _cy_extract_stress_marks(ipa)
    ipa = unicodedata.normalize("NFKD", ipa)
    marks: list[tuple[int, str]] = []
    clean_idx = 0
    for ch in ipa:
        if ch == _PRIMARY:
            marks.append((clean_idx, "primary"))
        elif ch == _SECONDARY:
            marks.append((clean_idx, "secondary"))
        else:
            clean_idx += 1
    return marks
