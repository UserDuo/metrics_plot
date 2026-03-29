"""
Syntax Metrics (Grammatical Structure).

Scientific Motivation:
This module assesses whether the grammatical structure of the text is preserved.
Character perturbations can alter part-of-speech tags or dependency relations,
potentially changing the argument structure and meaning of the sentence.
"""

import spacy
import Levenshtein
import warnings
from scipy.spatial.distance import jensenshannon
from collections import Counter
import numpy as np

# Lazy loading of Spacy model
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading Spacy model 'en_core_web_sm'...")
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp

def pos_divergence(s: str, sp: str) -> float:
    """
    Calculates the Part-of-Speech Divergence (POSD).

    Definition:
        Let p_x(c) be the empirical distribution of POS tags c in text x.
        POSD(x, x~) = JSD(p_x || p_x~) = sqrt(0.5 * KL(p || m) + 0.5 * KL(q || m)).

    Role:
        Measures distributional shift of syntactic categories.
        Character perturbations that change word identity may alter POS patterns even if text length is unchanged.
        Range: [0, 1].

    Args:
        s: Original text sequence x.
        sp: Perturbed text sequence x~.

    Returns:
        float: The Jensen-Shannon divergence.
    """
    nlp = get_nlp()
    doc_s = nlp(s)
    doc_sp = nlp(sp)
    
    # Get POS counts
    pos_s = Counter([token.pos_ for token in doc_s])
    pos_sp = Counter([token.pos_ for token in doc_sp])
    
    # Union of all POS tags found
    all_tags = list(set(pos_s.keys()) | set(pos_sp.keys()))
    
    if not all_tags:
        return 0.0
        
    # Create probability distributions
    def get_dist(counter, tags):
        total = sum(counter.values())
        if total == 0: return np.zeros(len(tags))
        return np.array([counter[t] for t in tags]) / total
        
    p = get_dist(pos_s, all_tags)
    q = get_dist(pos_sp, all_tags)
    
    # Calculate JSD (base 2 for bit-like scaling, though result is [0,1])
    return jensenshannon(p, q, base=2)

def get_dependency_edges(text: str):
    nlp = get_nlp()
    doc = nlp(text)
    # Extract set of (head_text, child_text, dep_label)
    # Using text instead of index for robustness against small shifts
    edges = set()
    for token in doc:
        # Skip root or handle it
        if token.dep_ == "ROOT":
            edges.add(("ROOT", "ROOT", token.text, token.dep_))
        else:
            edges.add((token.head.text, token.text, token.dep_))
    return edges

def dependency_overlap_score(s: str, sp: str) -> float:
    """
    Calculates the Dependency Overlap Score (DOS).

    Definition:
        Let Dep(x) produce a set of dependency edges E_x.
        J_dep = |E_x ∩ E_x~| / |E_x ∪ E_x~|.
        DOS(x, x~) = 1 - J_dep.

    Role:
        Captures structural stability of the dependency graph under perturbations.
        High DOS indicates major changes in syntactic relations.
        Range: [0, 1].

    Args:
        s: Original text sequence x.
        sp: Perturbed text sequence x~.

    Returns:
        float: The dependency divergence (1 - overlap).
    """
    edges_s = get_dependency_edges(s)
    edges_sp = get_dependency_edges(sp)
    
    if not edges_s and not edges_sp:
        return 0.0
        
    intersection = len(edges_s.intersection(edges_sp))
    union = len(edges_s.union(edges_sp))
    
    if union == 0:
        return 0.0
    
    return 1.0 - (intersection / union)

def tree_depth_change(s: str, sp: str) -> float:
    """
    Calculates the Tree Depth Change (TDC).

    Definition:
        For a parse tree of sentence x, let D(x) be the maximum tree depth.
        TDC(x, x~) = D(x~) - D(x).

    Role:
        Highlights global syntactic complexity variations due to perturbation.
        Insertions or deletions that create nested clauses increase depth, while simplifications reduce it.
        Range: Integers (-inf, +inf).

    Args:
        s: Original text sequence x.
        sp: Perturbed text sequence x~.

    Returns:
        float: The change in tree depth.
    """
    nlp = get_nlp()
    
    def get_depth(text):
        doc = nlp(text)
        if not doc: return 0
        roots = [token for token in doc if token.head == token]
        if not roots: return 0
        
        # DFS for depth
        max_depth = 0
        for root in roots:
            stack = [(root, 1)]
            while stack:
                node, depth = stack.pop()
                max_depth = max(max_depth, depth)
                for child in node.children:
                    stack.append((child, depth + 1))
        return max_depth

    depth_s = get_depth(s)
    depth_sp = get_depth(sp)
    
    return float(depth_sp - depth_s)
