"""
IPA string normalisation utilities.

Strips suprasegmental markers (stress, length, linking) and applies
Unicode NFKD normalisation so that downstream distance computation
is stress-blind and encoding-consistent.

Stress preservation
-------------------
By default all suprasegmental markers are removed.  Pass
``preserve_stress=True`` or a :class:`CleanConfig` to selectively
retain stress, length, or liaison markers.

Usage::

    >>> from phone_similarity.clean_phones import clean_phones, CleanConfig
    >>> clean_phones("ˈhɛloʊ")        # "hɛloʊ"  -- default, strips all
    >>> clean_phones("ˈhɛloʊ", preserve_stress=True)  # "ˈhɛloʊ"
    >>> cfg = CleanConfig(strip_stress=False, strip_length=True)
    >>> clean_phones("ˈhɛːloʊ", config=cfg)           # "ˈhɛloʊ"
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata

# ---------------------------------------------------------------------------
# Marker character sets
# ---------------------------------------------------------------------------
_STRESS_CHARS = "ˈˌ"  # primary + secondary stress
_LENGTH_CHARS = "ːˑ"  # long + half-long
_LIAISON_CHARS = "‿"  # liaison / linking

# Pre-compiled patterns for each category
_PAT_STRESS = re.compile(f"[{re.escape(_STRESS_CHARS)}]")
_PAT_LENGTH = re.compile(f"[{re.escape(_LENGTH_CHARS)}]")
_PAT_LIAISON = re.compile(f"[{re.escape(_LIAISON_CHARS)}]")

# Legacy "strip everything" pattern
_STRIP_PATTERN = re.compile(r"[ˈˌːˑ‿]")


# ---------------------------------------------------------------------------
# CleanConfig
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class CleanConfig:
    """Per-language normalisation settings.

    Controls which suprasegmental categories are stripped from IPA input.
    Any combination is valid; all default to ``True`` (strip everything)
    for backward compatibility.

    Attributes
    ----------
    strip_stress : bool
        Remove primary ``ˈ`` and secondary ``ˌ`` stress markers.
    strip_length : bool
        Remove long ``ː`` and half-long ``ˑ`` markers.
    strip_liaison : bool
        Remove linking ``‿`` markers.
    nfkd : bool
        Apply NFKD Unicode normalisation.

    Examples
    --------
    Finnish (length is contrastive, stress is not):

    >>> CleanConfig(strip_stress=True, strip_length=False)
    CleanConfig(strip_stress=True, strip_length=False, strip_liaison=True, nfkd=True)
    """

    strip_stress: bool = True
    strip_length: bool = True
    strip_liaison: bool = True
    nfkd: bool = True

    @property
    def strip_all(self) -> bool:
        """True when every suprasegmental category is being stripped."""
        return self.strip_stress and self.strip_length and self.strip_liaison


# ---------------------------------------------------------------------------
# Pre-built configs
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def clean_phones(
    ipa: str,
    *,
    preserve_stress: bool = False,
    config: CleanConfig | None = None,
) -> str:
    """Remove suprasegmental markers and NFKD-normalise an IPA string.

    Parameters
    ----------
    ipa : str
        Raw IPA transcription, possibly containing suprasegmental markers.
    preserve_stress : bool
        Convenience flag.  When ``True``, stress markers ``ˈˌ`` are kept.
        Ignored if *config* is supplied.
    config : CleanConfig, optional
        Fine-grained control over which categories to strip.  Overrides
        *preserve_stress* when given.

    Returns
    -------
    str
        Normalised IPA string.
    """
    # Resolve config --------------------------------------------------------
    if config is not None:
        cfg = config
    elif preserve_stress:
        cfg = PRESERVE_STRESS
    else:
        cfg = STRIP_ALL

    # NFKD normalisation ----------------------------------------------------
    if cfg.nfkd:
        ipa = unicodedata.normalize("NFKD", ipa)

    # Fast path: strip everything (most common case) -----------------------
    if cfg.strip_all:
        return _STRIP_PATTERN.sub("", ipa)

    # Selective stripping ---------------------------------------------------
    if cfg.strip_stress:
        ipa = _PAT_STRESS.sub("", ipa)
    if cfg.strip_length:
        ipa = _PAT_LENGTH.sub("", ipa)
    if cfg.strip_liaison:
        ipa = _PAT_LIAISON.sub("", ipa)
    return ipa


# ---------------------------------------------------------------------------
# Stress extraction helpers
# ---------------------------------------------------------------------------
_PRIMARY = "ˈ"
_SECONDARY = "ˌ"


def extract_stress_marks(ipa: str) -> list[tuple[int, str]]:
    """Find stress marker positions in an IPA string.

    Scans the (NFKD-normalised) IPA string and returns a list of
    ``(position, marker)`` pairs where *position* is the index in the
    **cleaned** string (markers excluded) of the segment immediately
    following the marker.

    Parameters
    ----------
    ipa : str
        IPA string that may contain stress markers.

    Returns
    -------
    list of (int, str)
        Each entry is ``(clean_index, marker)`` where *marker* is
        ``"primary"`` or ``"secondary"``.

    Examples
    --------
    >>> extract_stress_marks("ˈhɛˌloʊ")
    [(0, 'primary'), (2, 'secondary')]
    """
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
