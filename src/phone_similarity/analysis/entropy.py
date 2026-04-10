import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phone_similarity.bit_array_specification import BitArraySpecification


from bitarray import frozenbitarray


@dataclass
class SyllableEncoding:
    """Container for phoneme bitarray representation."""

    syllables: list[tuple[frozenbitarray, frozenbitarray, frozenbitarray]]

    def to_tuple(
        self,
    ) -> tuple[tuple[frozenbitarray, frozenbitarray, frozenbitarray], ...]:
        """Convert to tuple for hashing."""
        return tuple(self.syllables)


@dataclass
class EntropyMetrics:
    """Container for entropy analysis results."""

    total_words: int
    unique_patterns: int

    onset_entropy: float
    nucleus_entropy: float
    coda_entropy: float

    joint_entropy: float
    onset_utilization: float
    nucleus_utilization: float
    coda_utilization: float

    unique_onset_patterns: int
    unique_nucleus_patterns: int
    unique_coda_patterns: int


class PhonemeEntropyAnalyzer:
    """
    Analyzes entropy and bit utilization for phonemic bitarray representations.
    """

    SyllableEncoding = SyllableEncoding

    def __init__(self, spec: "BitArraySpecification"):
        """
        Initialize analyzer with a BitArraySpecification.

        Args:
            spec: The specification for bit array representations.
        """
        self.spec = spec
        self.onset_bits = len(spec.empty_vector("consonant"))
        self.nucleus_bits = len(spec.empty_vector("vowel"))
        self.coda_bits = len(spec.empty_vector("consonant"))

        # Maximum possible unique patterns for each component
        self.max_onset = 2**self.onset_bits
        self.max_nucleus = 2**self.nucleus_bits
        self.max_coda = 2**self.coda_bits

        # Storage for observed patterns
        self.onset_counts: Counter = Counter()
        self.nucleus_counts: Counter = Counter()
        self.coda_counts: Counter = Counter()
        self.joint_counts: Counter = Counter()

    def add_word(self, encoding: SyllableEncoding, frequency: int = 1) -> None:
        """
        Add a word's phoneme encoding to the analysis.

        Args:
            encoding: PhonemeEncoding with onset, nucleus, coda bitarrays
            frequency: Word frequency (for weighted entropy calculation)
        """
        for onset, nucleus, coda in encoding.syllables:
            self.onset_counts[onset] += frequency
            self.nucleus_counts[nucleus] += frequency
            self.coda_counts[coda] += frequency
        self.joint_counts[encoding.to_tuple()] += frequency

    def calculate_entropy(self, counts: Counter) -> float:
        """
        Calculate Shannon entropy for a distribution.

        Args:
            counts: Counter with pattern frequencies

        Returns:
            Shannon entropy in bits
        """
        total = sum(counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def calculate_utilization(self, unique_patterns: int, max_patterns: int) -> float:
        """
        Calculate what percentage of available bit patterns are actually used.

        Args:
            unique_patterns: Number of unique patterns observed
            max_patterns: Maximum possible patterns (2^bits)

        Returns:
            Utilization as a fraction (0.0 to 1.0)
        """
        return unique_patterns / max_patterns if max_patterns > 0 else 0.0

    def get_entropy_metrics(self) -> EntropyMetrics:
        """
        Calculate comprehensive entropy and utilization metrics.

        Returns:
            EntropyMetrics containing all analysis results
        """
        total_words = sum(self.joint_counts.values())
        unique_patterns = len(self.joint_counts)

        unique_onset = len(self.onset_counts)
        unique_nucleus = len(self.nucleus_counts)
        unique_coda = len(self.coda_counts)

        return EntropyMetrics(
            total_words=total_words,
            unique_patterns=unique_patterns,
            onset_entropy=self.calculate_entropy(self.onset_counts),
            nucleus_entropy=self.calculate_entropy(self.nucleus_counts),
            coda_entropy=self.calculate_entropy(self.coda_counts),
            joint_entropy=self.calculate_entropy(self.joint_counts),
            onset_utilization=self.calculate_utilization(unique_onset, self.max_onset),
            nucleus_utilization=self.calculate_utilization(unique_nucleus, self.max_nucleus),
            coda_utilization=self.calculate_utilization(unique_coda, self.max_coda),
            unique_onset_patterns=unique_onset,
            unique_nucleus_patterns=unique_nucleus,
            unique_coda_patterns=unique_coda,
        )

    def get_top_patterns(self, n: int = 20) -> dict[str, list[tuple[int, int]]]:
        """
        Get the most common patterns for each component.

        Args:
            n: Number of top patterns to return

        Returns:
            Dictionary with 'onset', 'nucleus', 'coda' keys mapping to
            lists of (pattern, count) tuples
        """
        return {
            "onset": self.onset_counts.most_common(n),
            "nucleus": self.nucleus_counts.most_common(n),
            "coda": self.coda_counts.most_common(n),
            "joint": self.joint_counts.most_common(n),
        }

    def print_report(self) -> None:
        """Print a comprehensive analysis report."""
        metrics = self.get_entropy_metrics()

        print("=" * 70)
        print("SYLLABLE BITARRAY ENTROPY ANALYSIS")
        print("=" * 70)
        print(
            f"\nBit Allocation: Onset={self.onset_bits}, Nucleus={self.nucleus_bits}, Coda={self.coda_bits}"
        )
        print(f"Total Words Analyzed: {metrics.total_words:,}")
        print(f"Unique Phoneme Patterns: {metrics.unique_patterns:,}")

        print("\n" + "-" * 70)
        print("ENTROPY MEASURES (in bits)")
        print("-" * 70)
        print(
            f"Onset Entropy:    {metrics.onset_entropy:.3f} / {self.onset_bits} bits ({metrics.onset_entropy / self.onset_bits * 100:.1f}% of max)"
        )
        print(
            f"Nucleus Entropy:  {metrics.nucleus_entropy:.3f} / {self.nucleus_bits} bits ({metrics.nucleus_entropy / self.nucleus_bits * 100:.1f}% of max)"
        )
        print(
            f"Coda Entropy:     {metrics.coda_entropy:.3f} / {self.coda_bits} bits ({metrics.coda_entropy / self.coda_bits * 100:.1f}% of max)"
        )
        print(f"Joint Entropy:    {metrics.joint_entropy:.3f} bits")

        redundancy = (
            metrics.onset_entropy + metrics.nucleus_entropy + metrics.coda_entropy
        ) - metrics.joint_entropy
        print(f"Redundancy:       {redundancy:.3f} bits")

        print("\n" + "-" * 70)
        print("BIT UTILIZATION")
        print("-" * 70)
        print(
            f"Onset:   {metrics.unique_onset_patterns:4d} / {self.max_onset:4d} patterns ({metrics.onset_utilization * 100:.1f}%)"
        )
        print(
            f"Nucleus: {metrics.unique_nucleus_patterns:4d} / {self.max_nucleus:4d} patterns ({metrics.nucleus_utilization * 100:.1f}%)"
        )
        print(
            f"Coda:    {metrics.unique_coda_patterns:4d} / {self.max_coda:4d} patterns ({metrics.coda_utilization * 100:.1f}%)"
        )

        # Interpretation
        print("\n" + "-" * 70)
        print("INTERPRETATION")
        print("-" * 70)

        if metrics.onset_utilization < 0.25:
            print("⚠ Onset is OVER-PROVISIONED (using <25% of available patterns)")
        elif metrics.onset_utilization > 0.75:
            print("⚠ Onset may be UNDER-PROVISIONED (using >75% of available patterns)")
        else:
            print("✓ Onset bit allocation appears appropriate")

        if metrics.nucleus_utilization < 0.25:
            print("⚠ Nucleus is OVER-PROVISIONED (using <25% of available patterns)")
        elif metrics.nucleus_utilization > 0.75:
            print("⚠ Nucleus may be UNDER-PROVISIONED (using >75% of available patterns)")
        else:
            print("✓ Nucleus bit allocation appears appropriate")

        if metrics.coda_utilization < 0.25:
            print("⚠ Coda is OVER-PROVISIONED (using <25% of available patterns)")
        elif metrics.coda_utilization > 0.75:
            print("⚠ Coda may be UNDER-PROVISIONED (using >75% of available patterns)")
        else:
            print("✓ Coda bit allocation appears appropriate")

        print("=" * 70)
