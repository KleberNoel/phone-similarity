import argparse
import csv
import logging
import os
from pathlib import Path

import requests

from phone_similarity.g2p.charsiu import LANGUAGE_CODES_CHARSIU

logger = logging.getLogger(__name__)

# Default cache directory for downloaded dictionaries.
# Dictionaries are downloaded on demand from CharsiuG2P GitHub and cached here.
DICT_CACHE_DIR = Path(os.path.expanduser("~/.cache/phono-sim/dicts"))

CHARSIU_DICT_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/{lang_code}.tsv"
)


def load_dictionary_tsv(lang_code: str, folder: Path | None = None) -> dict[str, str]:
    """
    Load a Charsiu TSV dictionary, downloading it on first use.

    Dictionaries are cached in ``~/.cache/phono-sim/dicts/`` (or *folder* if
    given).  If the TSV file does not exist locally it is fetched from the
    CharsiuG2P GitHub repository.

    Parameters
    ----------
    lang_code : str
        The language code for the dictionary (e.g., ``'eng-us'``, ``'fra'``).
    folder : Path, optional
        Override the cache directory.  Defaults to
        ``~/.cache/phono-sim/dicts/``.

    Returns
    -------
    Dict[str, str]
        A dictionary mapping words to their phonetic transcriptions.
        Example: ``{'word': 'p i tʰ', ...}``.
    """
    if lang_code not in LANGUAGE_CODES_CHARSIU:
        raise ValueError(f"{lang_code} must be in {LANGUAGE_CODES_CHARSIU}")

    if folder is None:
        folder = DICT_CACHE_DIR

    url = CHARSIU_DICT_URL_TEMPLATE.format(lang_code=lang_code)
    local_path = folder / f"{lang_code}.tsv"

    if not local_path.exists():
        logger.info("Downloading dictionary for %s from %s", lang_code, url)
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            os.makedirs(folder, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(r.content)
        except requests.RequestException as exc:
            logger.error("Failed to download %s: %s", url, exc)
            return {}

    dict_map: dict[str, str] = {}
    with open(local_path, encoding="utf-8") as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")
        for row in reader:
            if len(row) < 2 or "CONSONANTUNACCOUNTEDFOR" in row[1] or "QNOU" in row[1]:
                continue
            word = row[0].strip().lower()
            phones = row[1].strip()
            dict_map[word] = phones

    return dict_map


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lang", "-l", required=True)
    args = p.parse_args()

    if args.lang:
        d = load_dictionary_tsv(args.lang)
        print(f"Loaded {len(d)} entries for {args.lang}")
    else:
        raise ValueError("Problem with language arg (it should be --lang)")
