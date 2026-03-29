"""
Rhythm Metrics (Prosodic Structure).

Scientific Motivation:
This module quantifies the disruption of the text's prosodic and rhythmic structure.
Character perturbations can alter syllable counts, stress patterns, and the flow of speech,
which is critical for applications involving speech synthesis, poetry, or marketing copy.
"""

import Levenshtein
from g2p_en import G2p
import nltk

# Initialize G2p (heavy, maybe should be in ModelManager but it's small enough for now or global lazy)
_g2p = None

def ensure_nltk_resources():
    """Ensure necessary NLTK resources are downloaded."""
    resources = ['averaged_perceptron_tagger_eng', 'cmudict']
    for res in resources:
        try:
            nltk.data.find(f'help/taggers/{res}' if 'tagger' in res else f'corpora/{res}')
        except LookupError:
            try:
                # Try finding in standard paths if help/ path fails or just rely on download
                if 'tagger' in res:
                    nltk.data.find(f'taggers/{res}')
                else:
                    nltk.data.find(f'corpora/{res}')
            except LookupError:
                print(f"Downloading NLTK resource: {res}...")
                nltk.download(res, quiet=True)

def get_g2p():
    global _g2p
    if _g2p is None:
        ensure_nltk_resources()
        _g2p = G2p()
    return _g2p

def get_stress_pattern(text: str):
    g2p = get_g2p()
    out = g2p(text)
    # Filter for stress numbers (0, 1, 2)
    stresses = [ch for ph in out if (len(ph) > 0 and ph[-1].isdigit()) for ch in ph if ch.isdigit()]
    return "".join(stresses)

def syllable_count_change(s: str, sp: str) -> float:
    """
    Calculates the Syllable Count Change (SCC).

    Definition:
        Let Syl(x) return the total number of syllables.
        SCC(x, x~) = (Syl(x~) - Syl(x)) / Syl(x).

    Role:
        Tracks global rhythmic length in spoken form.
        Preserves meaning but spelling changes can still perturb syllable counts.
        Range: (-1, +inf).

    Args:
        s: Original text sequence x.
        sp: Perturbed text sequence x~.

    Returns:
        float: The relative change in syllable count.
    """
    stress_s = get_stress_pattern(s)
    stress_sp = get_stress_pattern(sp)
    
    syll_s = len(stress_s)
    syll_sp = len(stress_sp)
    
    if syll_s == 0:
        return 0.0
        
    return (syll_sp - syll_s) / syll_s

def stress_pattern_divergence(s: str, sp: str) -> float:
    """
    Calculates the Stress Pattern Divergence (SPD).

    Definition:
        Let Stress(x) be the sequence of stress symbols (0, 1, 2).
        SPD(x, x~) = d_stress(Stress(x), Stress(x~)) / max(|Stress(x)|, |Stress(x~)|).

    Role:
        Sensitive to metrical and accentual changes.
        A single syllable shift can disrupt regular rhythmic patterns in verse or slogans.
        Range: [0, 1].

    Args:
        s: Original text sequence x.
        sp: Perturbed text sequence x~.

    Returns:
        float: The normalized stress pattern edit distance.
    """
    stress_s = get_stress_pattern(s)
    stress_sp = get_stress_pattern(sp)
    
    if not stress_s and not stress_sp:
        return 0.0
        
    if not stress_s: # sp is not empty
        return 1.0 

    dist = Levenshtein.distance(stress_s, stress_sp)
    denom = max(len(stress_s), len(stress_sp))
    
    if denom == 0:
        return 0.0
        
    return dist / denom

def phonemic_levenshtein(s: str, sp: str) -> float:
    """
    Calculates the normalized Levenshtein distance on phonemes.
    Formula: LD_n(Ph(S), Ph(S')) = Lev(Ph(S), Ph(S')) / max(|Ph(S)|, |Ph(S')|)
    """
    g2p = get_g2p()
    # g2p returns a list of phonemes and punctuation
    # We filter for only alphanumeric-like phonemes (ignoring spaces if any)
    ph_s = [p for p in g2p(s) if p != ' ']
    ph_sp = [p for p in g2p(sp) if p != ' ']
    
    if not ph_s and not ph_sp:
        return 0.0
        
    # Map unique phonemes to characters for standard Levenshtein
    unique_phonemes = list(set(ph_s + ph_sp))
    p_map = {p: chr(i + 200) for i, p in enumerate(unique_phonemes)} # Start at 200 to avoid conflicts
    
    s_mapped = "".join([p_map[p] for p in ph_s])
    sp_mapped = "".join([p_map[p] for p in ph_sp])
    
    return Levenshtein.distance(s_mapped, sp_mapped) / max(len(ph_s), len(ph_sp))

def prosodic_flow_disruption_index(s: str, sp: str, gamma: float = 1.0, delta: float = 1.0) -> float:
    def segments_standard(x: str):
        return [len(seg) for seg in x.split(' ')]
    def segments_with_unicode(x: str):
        lengths = []
        cur = 0
        for ch in x:
            if ch == ' ':
                lengths.append(cur)
                cur = 0
            elif ch.isspace() and ch != ' ':
                lengths.append(cur)
                lengths.append(0)
                cur = 0
            else:
                cur += 1
        lengths.append(cur)
        return lengths
    L_o = segments_standard(s)
    L_d = segments_with_unicode(sp)
    pc = abs(len(L_d) - len(L_o))
    def cv(arr):
        if not arr:
            return 0.0
        m = sum(arr) / len(arr)
        if m == 0:
            return 0.0
        var = sum((x - m) ** 2 for x in arr) / len(arr)
        return (var ** 0.5) / m
    cv_delta = abs(cv(L_d) - cv(L_o))
    return gamma * pc + delta * cv_delta
