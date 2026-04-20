"""Cross-language cognate distance tests.

Tests that cognate word pairs (e.g. *father/père/vater/vader/padre*) are
more similar to each other than to unrelated words, using the feature-
weighted edit distance API from ``phone_similarity.distance_class``.
"""

from itertools import combinations

import pytest

pytestmark = pytest.mark.slow

from phone_similarity.bit_array_specification import BitArraySpecification
from phone_similarity.cross_language import compare_cross_language
from phone_similarity.distance_class import Distance
from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator
from phone_similarity.language import LANGUAGES
from phone_similarity.primitives import normalised_feature_edit_distance
from phone_similarity.universal_features import UniversalFeatureEncoder

# Fixtures

# Charsiu dict code -> language module name
_LANG_MAP: dict[str, str] = {
    "eng-us": "eng_us",
    "fra": "fra",
    "ger": "ger",
    "dut": "dut",
    "spa": "spa",
}

# Cognate sets: tuple of (eng-us, fra, ger, dut, spa)
COGNATES: dict[str, tuple[str, ...]] = {
    "father": ("father", "père", "vater", "vader", "padre"),
    "mother": ("mother", "mère", "mutter", "moeder", "madre"),
    "sun": ("sun", "soleil", "sonne", "zon", "sol"),
    "moon": ("moon", "lune", "mond", "maan", "luna"),
    "water": ("water", "eau", "wasser", "water", "agua"),
    "fire": ("fire", "feu", "feuer", "vuur", "fuego"),
}

LANG_CODES = list(_LANG_MAP.keys())


def _make_spec(module_name: str) -> BitArraySpecification:
    lang = LANGUAGES[module_name]
    consonants = set(lang.PHONEME_FEATURES.keys()) - lang.VOWELS_SET
    return BitArraySpecification(
        vowels=lang.VOWELS_SET,
        consonants=consonants,
        features=lang.FEATURES,
        features_per_phoneme=lang.PHONEME_FEATURES,
    )


@pytest.fixture(scope="module")
def g2ps() -> dict[str, CharsiuGraphemeToPhonemeGenerator]:
    return {lc: CharsiuGraphemeToPhonemeGenerator(lc) for lc in LANG_CODES}


@pytest.fixture(scope="module")
def specs() -> dict[str, BitArraySpecification]:
    return {lc: _make_spec(_LANG_MAP[lc]) for lc in LANG_CODES}


@pytest.fixture(scope="module")
def features_by_lang() -> dict[str, dict]:
    return {lc: LANGUAGES[_LANG_MAP[lc]].PHONEME_FEATURES for lc in LANG_CODES}


@pytest.fixture(scope="module")
def cognate_ipa(g2ps) -> dict[str, dict[str, str]]:
    """For each cognate set, look up IPA from the dictionaries.

    Returns ``{concept: {lang_code: ipa_string}}``.
    Skips any word not found in the dictionary.
    """
    result: dict[str, dict[str, str]] = {}
    for concept, words in COGNATES.items():
        ipa_by_lang: dict[str, str] = {}
        for lc, word in zip(LANG_CODES, words, strict=False):
            pron = g2ps[lc].pdict.get(word)
            if pron:
                # Take first pronunciation, strip stress markers for cleaner comparison
                ipa_by_lang[lc] = pron.split(",")[0].strip()
        if len(ipa_by_lang) >= 2:
            result[concept] = ipa_by_lang
    return result


# Tests


class TestCognateDistances:
    """Verify that the distance API produces meaningful results on cognates."""

    def test_all_cognate_sets_have_pronunciations(self, cognate_ipa):
        """Every cognate set should have at least 2 languages with pronunciations."""
        for concept, ipa_map in cognate_ipa.items():
            assert len(ipa_map) >= 2, f"Cognate '{concept}' has <2 languages with pronunciations"

    def test_edit_distance_same_language_is_zero(self, specs, features_by_lang):
        """Comparing a pronunciation with itself should give distance 0."""
        for lc in LANG_CODES:
            mod = _LANG_MAP[lc]
            LANGUAGES[mod]
            spec = specs[lc]
            feats = features_by_lang[lc]
            d = Distance(spec, feats)
            # pick any IPA string — use a known simple word
            ipa = "pa"
            assert d.normalised_edit_distance(ipa, ipa) == 0.0

    def test_cognate_pairs_have_finite_distance(self, cognate_ipa, specs, features_by_lang):
        """All cognate language pairs should produce a finite distance."""
        for concept, ipa_map in cognate_ipa.items():
            langs = sorted(ipa_map)
            for la, lb in combinations(langs, 2):
                merged_feats = UniversalFeatureEncoder.merge_inventories(
                    features_by_lang[la],
                    features_by_lang[lb],
                )
                tokens_a = specs[la].ipa_tokenizer(ipa_map[la])
                tokens_b = specs[lb].ipa_tokenizer(ipa_map[lb])
                d = normalised_feature_edit_distance(tokens_a, tokens_b, merged_feats)
                assert 0.0 <= d <= 1.0, f"{concept} ({la} vs {lb}): distance {d} out of range"

    def test_same_family_closer_than_cross_family(self, cognate_ipa, specs, features_by_lang):
        """Germanic pairs should (on average) be closer than Germanic-Romance.

        eng-us / ger / dut are Germanic.  fra / spa are Romance.
        We expect intra-family mean distance < inter-family mean distance
        across all cognate sets that have enough data.
        """
        germanic = {"eng-us", "ger", "dut"}
        romance = {"fra", "spa"}

        intra_germanic_dists = []
        cross_family_dists = []

        for _concept, ipa_map in cognate_ipa.items():
            langs = sorted(ipa_map)
            for la, lb in combinations(langs, 2):
                merged_feats = UniversalFeatureEncoder.merge_inventories(
                    features_by_lang[la],
                    features_by_lang[lb],
                )
                tokens_a = specs[la].ipa_tokenizer(ipa_map[la])
                tokens_b = specs[lb].ipa_tokenizer(ipa_map[lb])
                d = normalised_feature_edit_distance(tokens_a, tokens_b, merged_feats)

                if la in germanic and lb in germanic:
                    intra_germanic_dists.append(d)
                elif (la in germanic and lb in romance) or (la in romance and lb in germanic):
                    cross_family_dists.append(d)

        assert len(intra_germanic_dists) > 0, "No intra-Germanic pairs found"
        assert len(cross_family_dists) > 0, "No cross-family pairs found"

        mean_intra = sum(intra_germanic_dists) / len(intra_germanic_dists)
        mean_cross = sum(cross_family_dists) / len(cross_family_dists)

        # Germanic cognates should be more similar on average
        assert mean_intra < mean_cross, (
            f"Expected intra-Germanic ({mean_intra:.3f}) < cross-family ({mean_cross:.3f})"
        )

    def test_compare_cross_language_returns_all_pairs(self, cognate_ipa, specs, features_by_lang):
        """``compare_cross_language`` should return one entry per unordered pair."""
        for concept, ipa_map in cognate_ipa.items():
            results = compare_cross_language(ipa_map, specs, features_by_lang, metric="edit")
            n_langs = len(ipa_map)
            expected_pairs = n_langs * (n_langs - 1) // 2
            assert len(results) == expected_pairs, (
                f"{concept}: expected {expected_pairs} pairs, got {len(results)}"
            )
            for (_la, _lb), dist in results.items():
                assert 0.0 <= dist <= 1.0

    def test_pairwise_edit_distance_matrix_symmetric(self, cognate_ipa, specs, features_by_lang):
        """Pairwise edit distance matrix should be symmetric with zero diagonal."""
        # Use "father" cognate set
        ipa_map = cognate_ipa.get("father", {})
        if len(ipa_map) < 2:
            pytest.skip("Not enough father pronunciations")

        # Build a merged spec + features for all languages in this set
        all_feats = UniversalFeatureEncoder.merge_inventories(
            *(features_by_lang[lc] for lc in ipa_map),
        )

        # Pick one spec (we only need the tokenizer)
        first_lc = next(iter(ipa_map))
        spec = specs[first_lc]
        d = Distance(spec, all_feats)

        ipa_strings = list(ipa_map.values())
        matrix = d.pairwise_edit_distance(ipa_strings)

        n = len(ipa_strings)
        for i in range(n):
            assert matrix[i][i] == pytest.approx(0.0), "Diagonal should be 0"
            for j in range(i + 1, n):
                assert matrix[i][j] == pytest.approx(matrix[j][i]), (
                    f"Matrix not symmetric at [{i}][{j}]"
                )
