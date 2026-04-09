"""
IPA string normalisation utilities.

Strips suprasegmental markers (stress, length, linking) and applies
Unicode NFKD normalisation so that downstream distance computation
is stress-blind and encoding-consistent.
"""

import re
import unicodedata

# Suprasegmental markers to strip:
#   ˈ  primary stress       ˌ  secondary stress  # noqa: RUF003
#   ː  long                 ˑ  half-long  # noqa: RUF003
#   ‿  linking (liaison)
_STRIP_PATTERN = re.compile(r"[ˈˌːˑ‿]")


def clean_phones(ipa: str) -> str:
    """Remove stress/length markers and NFKD-normalise an IPA string.

    Parameters
    ----------
    ipa : str
        Raw IPA transcription, possibly containing suprasegmental markers.

    Returns
    -------
    str
        Normalised IPA string with stress and length markers removed.
    """
    ipa = unicodedata.normalize("NFKD", ipa)
    return _STRIP_PATTERN.sub("", ipa)
