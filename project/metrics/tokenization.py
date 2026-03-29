"""
Tokenization Metrics (Model Interface Stability).

Scientific Motivation:
These metrics quantify how character-level perturbations affect the segmentation of text into tokens.
Subword tokenizers (BPE, WordPiece, etc.) are sensitive to orthographic changes.
A minor typo can cause a word to shatter into multiple subwords (fragmentation) or change
the token distribution significantly, potentially affecting downstream model performance.
"""

import Levenshtein
from project.utils.models import model_manager
import string
import difflib
import math
from collections import Counter

def get_tokens(text: str, tokenizer_name: str = None):
    # Pass category="tokenizer" to let ModelManager use config default
    tokenizer = model_manager.get_tokenizer(tokenizer_name, category="tokenizer")
    return tokenizer.tokenize(text)

def token_count_change(s: str, sp: str, tokenizer_name: str = None) -> float:
    """
    Calculates the Token Count Change (TCC).

    Definition:
        Let T(x) be the token sequence produced by a fixed tokenizer.
        TCC(x, x~) = (|T(x~)| - |T(x)|) / |T(x)|

    Role:
        Measures segmentation instability induced by character edits.
        Highlights cases where minor spelling changes explode token count in subword tokenizers.
        Range: (-1, +inf).

    Args:
        s: Original text sequence x.
        sp: Perturbed text sequence x~.
        tokenizer_name: Optional tokenizer identifier.

    Returns:
        float: The relative change in token count.
    """
    tok_s = get_tokens(s, tokenizer_name)
    tok_sp = get_tokens(sp, tokenizer_name)
    
    if len(tok_s) == 0:
        return 0.0
        
    return (len(tok_sp) - len(tok_s)) / len(tok_s)

def fragmentation_index(s: str, sp: str, tokenizer_name: str = None) -> float:
    """
    Calculates the Fragmentation Index (FI).

    Definition:
        For each text, define its fragmentation level as tokens per character: F(x) = |T(x)| / |x|.
        FI(x, x~) = F(x~) - F(x).

    Role:
        Quantifies how perturbations shatter words into subword pieces.
        Large positive FI indicates that previously common subwords become rare or out-of-vocabulary.
        Range: typically small, [-1/|x|, +inf).

    Args:
        s: Original text sequence x.
        sp: Perturbed text sequence x~.
        tokenizer_name: Optional tokenizer identifier.

    Returns:
        float: The change in fragmentation level.
    """
    if not s or not sp:
        return 0.0

    tok_s = get_tokens(s, tokenizer_name)
    tok_sp = get_tokens(sp, tokenizer_name)
    
    f_s = len(tok_s) / len(s)
    f_sp = len(tok_sp) / len(sp)
    
    return f_sp - f_s

def normalized_entropy_delta(s: str, sp: str, tokenizer_name: str = None) -> float:
    """
    Calculates the Normalized Entropy Delta (NED).

    Definition:
        Let p_x(v) be the empirical token distribution over a vocabulary V in text x.
        Its normalized Shannon entropy is H*(x) = - (1 / log|V|) * sum(p_x(v) * log(p_x(v))).
        NED(x, x~) = H*(x~) - H*(x).

    Role:
        Captures how perturbations change diversity of token usage.
        Random character noise often increases token entropy; systematic misspellings may decrease it.
        Range: [-1, 1].

    Args:
        s: Original text sequence x.
        sp: Perturbed text sequence x~.
        tokenizer_name: Optional tokenizer identifier.

    Returns:
        float: The normalized entropy delta.
    """
    tok_s = get_tokens(s, tokenizer_name)
    tok_sp = get_tokens(sp, tokenizer_name)
    
    tokenizer = model_manager.get_tokenizer(tokenizer_name, category="tokenizer")
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257 # Default GPT2
    
    def normalized_entropy(tokens):
        if not tokens:
            return 0.0
        n = len(tokens)
        cnt = Counter(tokens)
        # Shannon entropy
        h = -sum((c / n) * math.log(c / n) for c in cnt.values()) # Natural log
        # Normalize by log(|V|)
        # Note: If vocab_size is 1, log is 0. Avoid div by zero.
        norm = math.log(max(2, vocab_size))
        return h / norm

    h_s = normalized_entropy(tok_s)
    h_sp = normalized_entropy(tok_sp)
    
    return h_sp - h_s

def subword_fertility_change(s: str, sp: str, tokenizer_name: str = None) -> float:
    """
    Calculates the change in Subword Fertility (Subwords per Word).
    Higher fertility usually means more fragmentation/rare tokens.
    """
    words_s = s.split()
    words_sp = sp.split()
    
    if not words_s: return 0.0
    
    tok_s = get_tokens(s, tokenizer_name)
    tok_sp = get_tokens(sp, tokenizer_name)
    
    fertility_s = len(tok_s) / len(words_s)
    fertility_sp = len(tok_sp) / len(words_sp) if words_sp else 0
    
    return fertility_sp - fertility_s

def oov_rate_change(s: str, sp: str, tokenizer_name: str = None) -> float:
    """
    Calculates the change in Out-Of-Vocabulary (OOV) rate.
    We define "OOV-like" as actual [UNK] tokens or raw byte fallback tokens (e.g. <0x..>).
    """
    tokenizer = model_manager.get_tokenizer(tokenizer_name, category="tokenizer")
    tok_s = tokenizer.tokenize(s)
    tok_sp = tokenizer.tokenize(sp)
    
    unk_token = tokenizer.unk_token
    
    def is_oov(token):
        if unk_token and token == unk_token:
            return True
        if token.startswith("<0x") and token.endswith(">"):
            return True
        return False
        
    def calculate_rate(tokens):
        if not tokens: return 0.0
        oov_count = sum(1 for t in tokens if is_oov(t))
        return oov_count / len(tokens)

    return calculate_rate(tok_sp) - calculate_rate(tok_s)

def token_overlap_ratio(s: str, sp: str, tokenizer_name: str = None) -> float:
    """
    Calculates the token overlap ratio.
    Formula: TOR = |tok(S) ∩ tok(S')| / max(|tok(S)|, |tok(S')|)
    """
    tok_s = set(get_tokens(s, tokenizer_name))
    tok_sp = set(get_tokens(sp, tokenizer_name))
    
    if not tok_s and not tok_sp:
        return 0.0
        
    intersection = len(tok_s.intersection(tok_sp))
    return intersection / max(len(tok_s), len(tok_sp))

def token_jaccard_divergence(s: str, sp: str, tokenizer_name: str = None) -> float:
    """
    Calculates Jaccard divergence on token sets.
    Formula: 1 - |tok(S) ∩ tok(S')| / |tok(S) ∪ tok(S')|
    """
    tok_s = set(get_tokens(s, tokenizer_name))
    tok_sp = set(get_tokens(sp, tokenizer_name))
    if not tok_s and not tok_sp:
        return 0.0
    intersection = len(tok_s.intersection(tok_sp))
    union = len(tok_s.union(tok_sp))
    if union == 0:
        return 0.0
    return 1.0 - (intersection / union)

WHITESPACE_CHARS = {
    '\u0009', '\u000A', '\u000B', '\u000C', '\u000D',
    '\u0020', '\u0085', '\u00A0', '\u1680',
    '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', '\u200A',
    '\u2028', '\u2029', '\u202F', '\u205F', '\u3000', '\u200B'
}

def intra_word_whitespace_rate(s: str, sp: str) -> float:
    """
    Estimates the rate of whitespace insertions occurring inside alphanumeric word bodies.
    """
    def internal_ws_count(text: str) -> int:
        c = 0
        for i, ch in enumerate(text):
            if ch in WHITESPACE_CHARS:
                left = text[i-1] if i-1 >= 0 else ''
                right = text[i+1] if i+1 < len(text) else ''
                if (left.isalnum() and right.isalnum()):
                    c += 1
        return c
    base = internal_ws_count(s)
    cur = internal_ws_count(sp)
    diff = max(0, cur - base)
    denom = max(1, len(sp))
    return diff / denom

def token_boundary_edit_rate(s: str, sp: str, tokenizer_name: str = None) -> float:
    tok_s = get_tokens(s, tokenizer_name)
    tok_sp = get_tokens(sp, tokenizer_name)
    if not tok_s and not tok_sp:
        return 0.0
    sm = difflib.SequenceMatcher(a=tok_s, b=tok_sp)
    edits = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'insert':
            edits += (j2 - j1)
        elif tag == 'delete':
            edits += (i2 - i1)
    denom = max(len(tok_s), len(tok_sp))
    if denom == 0:
        return 0.0
    return edits / denom
