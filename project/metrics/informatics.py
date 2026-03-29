"""
Informatics Metrics (Surface-Level Information Theory).

Scientific Motivation:
This module quantifies perturbations as purely symbolic transformations, ignoring semantics.
It provides the baseline for "mechanical" degradation:
1. Edit Distance (Levenshtein): Represents the minimum human/machine effort to restore the text.
2. Information Density (Compression): Proxies for Kolmogorov complexity. 
   - A rise in compression size suggests the introduction of noise or randomness.
   - A drop suggests repetitive or simplified patterns (e.g., deletion of complex words).
"""

import gzip
import math
from collections import Counter
try:
    import Levenshtein  # type: ignore
    _HAS_LEV = True
except ImportError:
    _HAS_LEV = False
    import difflib

    def _lev_distance(a: str, b: str) -> int:
        if a == b: return 0
        if not a: return len(b)
        if not b: return len(a)
        n, m = len(a), len(b)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, m + 1):
                tmp = dp[j]
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = tmp
        return dp[m]

    def _editops(a: str, b: str):
        ops = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
            if tag == "replace": ops.append(("replace", i1, j1))
            elif tag == "delete": ops.append(("delete", i1, j1))
            elif tag == "insert": ops.append(("insert", i1, j1))
        return ops

def normalized_levenshtein(s: str, sp: str) -> float:
    """
    Calculates the Normalized Levenshtein Distance (NL).

    Definition:
        Let d_L(x, x~) be the Levenshtein edit distance between character sequences.
        NL(x, x~) = d_L(x, x~) / max(|x|, |x~|)

    Role:
        Directly measures how many characters were changed relative to length.
        Sensitive to local typos, transpositions, or adversarial character insertions.
        Range: [0, 1].

    Args:
        s: Original text sequence x.
        sp: Perturbed text sequence x~.

    Returns:
        float: The normalized distance.
    """
    if not s and not sp:
        return 0.0
    if _HAS_LEV:
        d = Levenshtein.distance(s, sp)
    else:
        d = _lev_distance(s, sp)
    return d / max(len(s), len(sp))

def char_ngram_jaccard(s: str, sp: str, n: int = 3) -> float:
    """
    Calculates the Character n-Gram Jaccard Distance (CNJ).

    Definition:
        For a fixed n, let G_n(x) be the set of all overlapping character n-grams in x.
        J_n(x, x~) = |G_n(x) ∩ G_n(x~)| / |G_n(x) ∪ G_n(x~)|
        CNJ(x, x~) = 1 - J_n(x, x~)

    Role:
        Captures local string pattern disruption.
        Sensitive to systematic pattern changes (e.g., consistent replacement of a character sequence).
        Range: [0, 1].

    Args:
        s: Original text sequence x.
        sp: Perturbed text sequence x~.
        n: n-gram length (default 3).

    Returns:
        float: The Jaccard distance.
    """
    def get_ngrams(text, n):
        return set(text[i:i+n] for i in range(len(text)-n+1))
    
    ng_s = get_ngrams(s, n)
    ng_sp = get_ngrams(sp, n)
    
    if not ng_s and not ng_sp:
        return 0.0
    
    intersection = len(ng_s.intersection(ng_sp))
    union = len(ng_s.union(ng_sp))
    
    if union == 0:
        return 0.0

    return 1.0 - (intersection / union)

def compression_delta(s: str, sp: str) -> float:
    """
    Calculates the Compression Delta (CD).

    Definition:
        Given a compressor C returning compressed length in bits (here bytes * 8 implied, relative).
        CD(x, x~) = (C(x~) - C(x)) / C(x)

    Role:
        Approximates change in algorithmic information content.
        Character perturbations that reduce redundancy usually decrease compressibility, yielding higher CD.
        Range: (-1, +inf).

    Args:
        s: Original text sequence x.
        sp: Perturbed text sequence x~.

    Returns:
        float: The normalized compression change.
    """
    c_s = len(gzip.compress(s.encode()))
    c_sp = len(gzip.compress(sp.encode()))
    
    if c_s == 0:
        return 0.0
        
    return (c_sp - c_s) / c_s

def edit_operation_counts(s: str, sp: str) -> dict:
    """
    Classifies edit operations into insertion, deletion, substitution.
    Returns counts of each operation type.
    """
    ops = Levenshtein.editops(s, sp) if _HAS_LEV else _editops(s, sp)
    counts = {"insert": 0, "delete": 0, "replace": 0}
    for op in ops:
        if op[0] in counts:
            counts[op[0]] += 1
    return counts
