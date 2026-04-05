import argparse
import csv
import os
from pathlib import Path
from typing import Dict

import requests

from phone_similarity.g2p.charsiu import LANGUAGE_CODES_CHARSIU


def load_dictionary_tsv(
    lang_code: str, folder: Path = Path(__file__).parents[3] / "dicts"
) -> Dict[str, str]:
    """
    Fetch or load a Charsiu TSV dictionary.

    This function fetches a TSV dictionary from a URL or loads it from a local
    file. The dictionary is expected to have two columns: word and phones.
    It returns a dictionary mapping words to their phonetic transcriptions.

    Parameters
    ----------
    lang_code : str
        The language code for the dictionary (e.g., 'eng', 'fra').
    folder : str, optional
        The folder to store the downloaded dictionary, by default "dicts".

    Returns
    -------
    Dict[str, str]
        A dictionary mapping words to their phonetic transcriptions.
        Example: { 'word': 'p i tʰ', ... }.
    """
    if lang_code not in LANGUAGE_CODES_CHARSIU:
        raise ValueError(f"{lang_code} must be in {LANGUAGE_CODES_CHARSIU}")

    url = f"https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/{lang_code}.tsv"
    local_path = folder / f"{lang_code}.tsv"

    if not local_path.exists():
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            os.makedirs(folder, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(r.content)
        except requests.HTTPError as http_exception:
            print(f"Failed to download {url}: {http_exception}")
            return {}

    dict_map = {}
    try:
        with open(local_path, "r", encoding="utf-8") as tsvfile:
            reader = csv.reader(tsvfile, delimiter="\t")
            for row in reader:
                if (
                    len(row) < 2
                    or "CONSONANTUNACCOUNTEDFOR" in row[1]
                    or "QNOU" in row[1]
                ):
                    continue
                word = row[0].strip().lower()
                phones = row[1].strip()
                dict_map[word] = phones
    except Exception as exception:
        raise exception

    return dict_map


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lang", "-l", required=True)
    args = p.parse_args()

    if args.lang:
        load_dictionary_tsv(args.lang)
    else:
        raise ValueError("Problem with language arg (it should be --lang)")
