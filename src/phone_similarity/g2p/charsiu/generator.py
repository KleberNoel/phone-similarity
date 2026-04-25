import logging
import math
import os
import pickle
import sys
from collections.abc import Sequence
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Union

from phone_similarity.g2p.charsiu import LANGUAGE_CODES_CHARSIU, load_dictionary


class ResearchUseOnlyError(RuntimeError):
    """Raised when neural G2P is used without explicit research opt-in."""


RESEARCH_OPT_IN_ENV_VAR = "PHONE_SIM_ALLOW_RESEARCH_G2P"


class GraphemeToPhonemeResourceType(Enum):
    DICT = 1
    G2P_GENERATOR = 2


class CharsiuGraphemeToPhonemeGenerator:
    """CharsiuGraphemeToPhonemeGenerator.

    A class that uses the Charsiu models defined in [1] to perform
    grapheme-to-phoneme conversion.

    Both the ONNX model and the pronunciation dictionary are loaded lazily —
    the dictionary on first access of ``.pdict``, and the model on first
    ``.generate()`` call.  This keeps ``import phone_similarity`` fast and
    avoids downloading resources that are never used.

    Parameters
    ----------
    language : str
        The language for which to generate phonemes. Must be a valid
        language code from `phone_similarity.g2p.charsiu.LANGUAGE_CODES_CHARSIU`.
    use_cache : bool
        Whether to pickle the parsed dictionary for faster subsequent loads.

    Attributes
    ----------
    pdict : Dict[str, str]
        A dictionary mapping words to their phonemic representations for the
        specified language.  Loaded lazily on first access.

    Methods
    -------
    generate(words: Tuple[str], **generation_kwargs)
        Generate phonemes for a list of words.
    get_phonemes_for_word(word, language, generation_kwargs)
        Get phonemes for a single word, first trying the dictionary.

    References
    ----------
    [1] J. Zhu, C. Zhang, and D. Jurgens,
        "Byt5 model for massively multilingual grapheme-to-phoneme conversion," 2022.
        [Online]. Available: https://arxiv.org/abs/2204.03067
        GitHub: https://github.com/lingjzhu/CharsiuG2P

    """

    DEFAULT_TOKENIZER_MODEL_NAME: str = "google/byt5-small"
    DEFAULT_ONNX_MODEL_NAME: str = "klebster/g2p_multilingual_byT5_tiny_onnx"

    def __init__(
        self,
        language: str,
        use_cache: bool = True,
        allow_research_g2p: bool = False,
    ):
        """Initializes the CharsiuGraphemeToPhonemeGenerator.

        Parameters
        ----------
        language : str
            The language for which to generate phonemes. Must be a valid
            language code from `phone_similarity.g2p.charsiu.LANGUAGE_CODES_CHARSIU`.
        use_cache : bool
            Whether to use a pickle cache for the phoneme dictionary.
        allow_research_g2p : bool
            Explicit opt-in for neural G2P generation.

            Neural generation with CharsiuG2P is intended for research use.
            By default this class allows dictionary lookup only. To enable
            model generation, either pass ``allow_research_g2p=True`` or set
            environment variable ``PHONE_SIM_ALLOW_RESEARCH_G2P=1``.
        """
        if language not in LANGUAGE_CODES_CHARSIU:
            raise ValueError(
                f"Unsupported language {language!r}. "
                f"Must be one of: {', '.join(sorted(LANGUAGE_CODES_CHARSIU))}"
            )

        self._language = language
        self._use_cache = use_cache
        self._allow_research_g2p = allow_research_g2p

        self._pdict: dict[str, str] | None = None
        self._model = None
        self._tokenizer = None

    def _research_g2p_enabled(self) -> bool:
        env_val = os.environ.get(RESEARCH_OPT_IN_ENV_VAR, "")
        env_enabled = env_val.strip().lower() in {"1", "true", "yes", "on"}
        return self._allow_research_g2p or env_enabled

    def _ensure_research_opt_in(self) -> None:
        if self._research_g2p_enabled():
            return
        raise ResearchUseOnlyError(
            "Neural G2P generation is disabled by default because CharsiuG2P "
            "is intended for research use and has uneven data quality across "
            "languages. To enable it explicitly, pass "
            "allow_research_g2p=True when constructing "
            "CharsiuGraphemeToPhonemeGenerator, or set "
            f"{RESEARCH_OPT_IN_ENV_VAR}=1."
        )

    def _ensure_dict_loaded(self) -> None:
        """Load the pronunciation dictionary on first use.

        The dictionary is downloaded from the CharsiuG2P GitHub repo (if
        not already cached) and optionally pickled for faster subsequent
        loads.
        """
        if self._pdict is not None:
            return

        cache_dir = Path(os.path.expanduser("~/.cache/phono-sim"))
        cache_dir.mkdir(exist_ok=True)
        py_version = f"py{sys.version_info.major}.{sys.version_info.minor}"
        cache_file = cache_dir / f"{self._language}_{py_version}.pkl"

        if self._use_cache and cache_file.exists():
            with open(cache_file, "rb") as f:
                self._pdict = pickle.load(f)
        else:
            self._pdict = load_dictionary.load_dictionary_tsv(self._language)
            if self._use_cache:
                with open(cache_file, "wb") as f:
                    pickle.dump(self._pdict, f)

    def _ensure_model_loaded(self) -> None:
        """Load the ONNX model and tokenizer on first use.

        This method is called automatically by `.generate()`. It defers
        the import of ``optimum.onnxruntime`` and ``transformers`` until
        inference is actually needed, keeping dictionary-only usage fast
        and lightweight.
        """
        if self._model is not None:
            return

        self._ensure_research_opt_in()

        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer

        self._model = ORTModelForSeq2SeqLM.from_pretrained(self.DEFAULT_ONNX_MODEL_NAME)
        self._tokenizer = AutoTokenizer.from_pretrained(self.DEFAULT_TOKENIZER_MODEL_NAME)

    @property
    def pdict(self) -> dict[str, str]:
        """A dictionary mapping words to their phonemic representations.

        The dictionary is loaded lazily on first access — it will be
        downloaded from CharsiuG2P GitHub if not already cached locally.

        Returns
        -------
        Dict[str, str]
            The phoneme dictionary mapping words to comma-separated
            pronunciations.
        """
        self._ensure_dict_loaded()
        if self._pdict is None:
            raise RuntimeError("Dictionary failed to load")
        return self._pdict

    @lru_cache(maxsize=2048)
    def generate(
        self, words: tuple[str], **generation_kwargs
    ) -> Union[list[str], list[list[str]]]:
        """Generate phonemes for a list of words.

        This method uses the Charsiu ONNX model to generate phonemic
        representations for a given list of words.

        Please note that while the original authors state
        "We do not find beam search helpful. Greedy decoding is enough."
        We expose beam search here to expand the number of possibilities

        Parameters
        ----------
        words : Tuple[str]
            A tuple of words to be converted to phonemes.
        **generation_kwargs : dict
            Additional keyword arguments to be passed to the underlying
            model's `generate` method.

        Returns
        -------
        List[Union[str, List[str]]]
            A list of phonemic representations for each word. If
            `num_return_sequences` is greater than 1, a list of tuples
            is returned, where each tuple contains multiple phonemic
            representations for a single word.
        """
        self._ensure_research_opt_in()
        self._ensure_model_loaded()

        _words = []
        for word in words:
            _words.append(
                f"<{self._language}>: {word}"
                if not word.startswith(f"<{self._language}>: ")
                else word
            )
        out = self._tokenizer(_words, padding=True, add_special_tokens=False, return_tensors="pt")

        num_return_sequences_active = bool("num_return_sequences" in generation_kwargs)

        preds = self._model.generate(**out, **generation_kwargs)
        if not isinstance(preds, dict) and not hasattr(preds, "sequences"):
            sequences_preds = preds
            sequences_probs = [1]
        else:
            sequences_preds = preds["sequences"] if isinstance(preds, dict) else preds.sequences
            if hasattr(preds, "sequences_scores") and preds.sequences_scores is not None:
                sequences_probs = [math.exp(float(sc)) for sc in preds.sequences_scores]
            else:
                sequences_probs = [1]

        if num_return_sequences_active:
            sequences_preds = sequences_preds.reshape(
                len(_words), generation_kwargs["num_return_sequences"], -1
            )
            phones = [
                tuple(self._tokenizer.batch_decode(pred.tolist(), skip_special_tokens=True))
                for pred in sequences_preds
            ]
            return phones, sequences_probs  # type: ignore

        phones = self._tokenizer.batch_decode(sequences_preds.tolist(), skip_special_tokens=True)
        return phones, sequences_probs  # type: ignore

    def generate_batched(
        self,
        words: Sequence[str],
        *,
        batch_size: int = 16,
        **generation_kwargs,
    ) -> tuple[list[Union[str, tuple[str, ...]]], list[float]]:
        """Generate phones for many words in fixed-size batches.

        This is a throughput-oriented wrapper over :meth:`generate` for
        large word lists. It preserves input order and aggregates per-batch
        outputs.

        Parameters
        ----------
        words : sequence of str
            Input words to decode.
        batch_size : int
            Number of words per model call (default 16).
        **generation_kwargs : dict
            Forwarded to :meth:`generate`.

        Returns
        -------
        tuple[list, list]
            ``(phones, probs)`` aggregated across all batches.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0 (got {batch_size})")

        word_list = list(words)
        if not word_list:
            return [], []

        all_phones: list[Union[str, tuple[str, ...]]] = []
        all_probs: list[float] = []

        for start in range(0, len(word_list), batch_size):
            batch_words = tuple(word_list[start : start + batch_size])
            phones, probs = self.generate(batch_words, **generation_kwargs)
            all_phones.extend(phones)
            all_probs.extend(float(p) for p in probs)

        return all_phones, all_probs

    def get_phones_from_dict(self, word: str) -> str:
        """get phonemes from a pronunciations dictionary"""
        maybe_phonemes = self.pdict.get(word, None)
        if maybe_phonemes is not None:
            return maybe_phonemes
        error_message = f"{word} was not found in the dictionary"
        logging.error(error_message)
        raise ValueError(error_message)

    @lru_cache(maxsize=2048)
    def get_phones_for_word(
        self,
        word: str,
        limit_resource: GraphemeToPhonemeResourceType | None,
        **generation_kwargs,
    ) -> tuple[Union[str, tuple[tuple[str]]]]:
        """Get phones for a single word.

        This method first attempts to look up the word in the phone
        dictionary. If the word is not found, it uses the Charsiu model available
        to generate the phone representation.

        Parameters
        ----------
        word : str
            The word to be converted to phones.
        limit_resource : Optional[GraphemeToPhonemeResourceType]
            limit the resource to use only either the G2P 'Generator', or the 'Dict'.
            If left as None, both will be used.
        **generation_kwargs :
            Additional keyword arguments to be passed to the underlying
            model's `generate` method.

        Returns
        -------
        Union[str, Tuple[str]]
            The phone representation of the word. If found in the
            dictionary, a list of phones is returned. Otherwise, a

            single string of phones is returned.
        """

        if limit_resource and limit_resource == GraphemeToPhonemeResourceType.DICT:
            phoneme_strings: list[str] = self.get_phones_from_dict(word).split()
            return tuple(phoneme_strings)  # type: ignore

        elif limit_resource and limit_resource == GraphemeToPhonemeResourceType.G2P_GENERATOR:
            phonemes = self.generate(tuple([word]), **generation_kwargs)  # type: ignore
            return tuple(phonemes) if isinstance(phonemes, list) else phonemes  # type: ignore

        generated = self.generate(tuple([word]), **generation_kwargs)
        dict_phones = self.get_phones_from_dict(word).split()
        # Combine model output with dictionary lookup without mutating
        # the cached generate() return value.
        if isinstance(generated, tuple) and len(generated) == 2:
            phones, probs = generated
            combined = list(phones) if isinstance(phones, list) else [phones]
            combined.append(dict_phones)
            return tuple(combined), probs  # type: ignore
        combined = list(generated) if isinstance(generated, list) else [generated]
        combined.append(dict_phones)
        return tuple(combined)  # type: ignore
