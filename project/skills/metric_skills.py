from project.skills.registry import registry
from project.metrics import informatics, tokenization, semantic, rhythm, visualization, syntax

def register_all_metrics():
    # Informatics
    registry.register(
        informatics.normalized_levenshtein,
        name="normalized_levenshtein",
        description="Calculates normalized Levenshtein distance between original string s and disturbed string sp. Returns float [0,1]."
    )
    registry.register(
        informatics.char_ngram_jaccard,
        name="char_ngram_jaccard",
        description="Calculates character n-gram Jaccard distance. Sensitive to local spelling changes."
    )
    registry.register(
        informatics.compression_delta,
        name="compression_delta",
        description="Calculates compression-based information delta. Measures complexity change."
    )
    registry.register(
        informatics.edit_operation_counts,
        name="edit_operation_counts",
        description="Classifies edit operations into insertion, deletion, substitution."
    )
    registry.register(
        tokenization.normalized_entropy_delta,
        name="normalized_entropy_delta",
        description="Normalized Entropy Delta combining character and word-level entropy changes."
    )

    # Tokenization
    registry.register(
        tokenization.token_count_change,
        name="token_count_change",
        description="Calculates relative change in token count."
    )
    registry.register(
        tokenization.fragmentation_index,
        name="fragmentation_index",
        description="Calculates Fragmentation Index (Edit distance on token list)."
    )

    # Semantic
    registry.register(
        semantic.lm_surprisal_delta,
        name="lm_surprisal_delta",
        description="Calculates delta in Language Model Surprisal (NLL/token)."
    )
    registry.register(
        semantic.semantic_entailment_score,
        name="semantic_entailment_score",
        description="Calculates semantic entailment probability using NLI. Measures logical consistency."
    )
    registry.register(
        semantic.contextual_embedding_distance,
        name="contextual_embedding_distance",
        description="Contextual embedding distance (cosine) capturing semantic displacement."
    )

    # Syntax
    registry.register(
        syntax.dependency_overlap_score,
        name="dependency_overlap_score",
        description="Calculates Jaccard similarity of dependency edges."
    )
    registry.register(
        syntax.tree_depth_change,
        name="tree_depth_change",
        description="Calculates relative change in dependency tree depth."
    )
    registry.register(
        syntax.pos_divergence,
        name="pos_divergence",
        description="Calculates Jensen-Shannon Divergence of POS tag distributions."
    )

    # Rhythm
    registry.register(
        rhythm.syllable_count_change,
        name="syllable_count_change",
        description="Calculates relative change in syllable count."
    )
    registry.register(
        rhythm.stress_pattern_divergence,
        name="stress_pattern_divergence",
        description="Calculates stress pattern divergence (prosody change)."
    )

    # Visualization
    registry.register(
        visualization.rendered_ssim_distance,
        name="ssim_distance", # Updated name to match study.py
        description="Calculates SSIM distance on rendered text images. Captures visual disturbances."
    )
    registry.register(
        visualization.glyph_layout_displacement,
        name="glyph_displacement", # Updated name to match study.py
        description="Calculates glyph layout displacement based on text width/layout."
    )
    registry.register(
        visualization.spatial_dispersion_salience_score,
        name="spatial_dispersion_salience_score",
        description="Spatial Dispersion & Salience Score based on center-of-mass and dispersion shifts."
    )


def calculate_composite_score(s: str, sp: str) -> dict:
    """
    Calculates the Composite Disturbance Score (Section 8.2).
    """
    # Weights from report
    weights = {
        "informatics": 0.15,
        "tokenization": 0.25,
        "semantics": 0.35,
        "rhythm": 0.10,
        "visualization": 0.15
    }
    
    # Calculate representative metrics for each category (simplified selection)
    # Informatics: Normalized Levenshtein
    inf = informatics.normalized_levenshtein(s, sp)
    
    # Tokenization: Fragmentation Index
    tok = tokenization.fragmentation_index(s, sp)
    
    # Semantics: Semantic Drift
    sem = semantic.embedding_cosine_drift(s, sp)
    
    # Rhythm: Stress Pattern Divergence & Phonemic Levenshtein (Average)
    rhy_stress = rhythm.stress_pattern_divergence(s, sp)
    rhy_phon = rhythm.phonemic_levenshtein(s, sp)
    rhy = (rhy_stress + rhy_phon) / 2
    
    # Syntax: Dependency Divergence (1 - overlap)
    syn_overlap = syntax.dependency_overlap_score(s, sp)
    syn = 1.0 - syn_overlap
    
    # Visualization: Rendered SSIM
    vis = visualization.rendered_ssim_distance(s, sp)
    
    # Updated weights to include Syntax
    # Normalizing weights to sum to 1 approx
    # Inf: 0.15, Tok: 0.20, Sem: 0.25, Syn: 0.15, Rhy: 0.10, Vis: 0.15
    weights = {
        "informatics": 0.15,
        "tokenization": 0.20,
        "semantics": 0.25,
        "syntax": 0.15,
        "rhythm": 0.10,
        "visualization": 0.15
    }
    
    composite = (
        weights["informatics"] * inf +
        weights["tokenization"] * tok +
        weights["semantics"] * sem +
        weights["syntax"] * syn +
        weights["rhythm"] * rhy +
        weights["visualization"] * vis
    )
    
    return {
        "composite_score": composite,
        "details": {
            "informatics": inf,
            "tokenization": tok,
            "semantics": sem,
            "syntax": syn,
            "rhythm": rhy,
            "visualization": vis
        }
    }

registry.register(
    calculate_composite_score,
    name="calculate_composite_score",
    description="Calculates the holistic Composite Disturbance Score combining all metric categories."
)
