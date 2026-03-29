"""
Semantic Metrics (Meaning Preservation).

Scientific Motivation:
This module quantifies the semantic stability of the text under perturbation.
It uses three complementary approaches:
1. Language Modeling (Surprisal): Measures how "unnatural" or improbable the perturbed text becomes.
2. Text Entailment (NLI): Measures whether the logical meaning is preserved (entailment vs. contradiction).
3. Embedding Space (Cosine): Measures the geometric drift in high-dimensional semantic space.
"""

import torch
import torch.nn.functional as F
from project.utils.models import model_manager

def get_sentence_embedding(text: str, model_name: str = None):
    # Pass category="embedding" to let ModelManager use config default
    model = model_manager.get_model(model_name, category="embedding")
    tokenizer = model_manager.get_tokenizer(model_name, category="embedding")
    device = model_manager.get_device()
    
    inp = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp)
    
    # Mean pooling
    return out.last_hidden_state.mean(dim=1).squeeze()

def contextual_embedding_distance(s: str, sp: str, model_name: str = None) -> float:
    """
    Calculates the Contextual Embedding Distance (CED).

    Definition:
        We compute sentence embeddings via an encoder E: e_x = E(x) and e_x~ = E(x~).
        CED(x, x~) = 1 - cos(e_x, e_x~).

    Role:
        Measures semantic drift in embedding space, capturing subtle meaning changes even when surface form is similar.
        Range: [0, 2], typically [0, 1].

    Args:
        s: Original text sequence x.
        sp: Perturbed text sequence x~.
        model_name: Optional model identifier.

    Returns:
        float: The embedding distance.
    """
    v1 = get_sentence_embedding(s, model_name)
    v2 = get_sentence_embedding(sp, model_name)
    sim = F.cosine_similarity(v1, v2, dim=0).item()
    return 1.0 - sim

def get_nll(text: str, model_name: str = None):
    model = model_manager.get_model(model_name, model_type="causal", category="language_model")
    tokenizer = model_manager.get_tokenizer(model_name, category="language_model")
    device = model_manager.get_device()
    
    enc = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**enc, labels=enc["input_ids"])
    
    # output.loss is the average NLL per token (CrossEntropyLoss default)
    # Returns (avg_nll, num_tokens)
    return output.loss.item(), len(enc["input_ids"][0])

def lm_surprisal_delta(s: str, sp: str, model_name: str = None) -> float:
    """
    Calculates the Language Model Surprisal Delta (LMSD).

    Definition:
        Let L assign token probabilities. S(x) is the average surprisal per token.
        LMSD(x, x~) = S(x~) - S(x).

    Role:
        Captures semantic and syntactic plausibility change from the model’s perspective.
        Small spelling edits can drastically increase surprisal even when human‑perceived meaning is stable.
        Positive values indicate x~ is less probable.
        Range: (-inf, +inf).

    Args:
        s: Original text sequence x.
        sp: Perturbed text sequence x~.
        model_name: Optional model identifier.

    Returns:
        float: The surprisal delta.
    """
    # get_nll returns average NLL per token, which IS the definition of S(x) in our paper.
    loss_s, _ = get_nll(s, model_name)
    loss_sp, _ = get_nll(sp, model_name)
    
    return loss_sp - loss_s

def semantic_entailment_score(s: str, sp: str, model_name: str = None) -> float:
    """
    Calculates the Semantic Entailment Score (SES).

    Definition:
        Using an NLI model, let p_ent(x => x~) be the entailment probability.
        SES(x, x~) = 1 - 0.5 * (p_ent(x => x~) + p_ent(x~ => x)).

    Role:
        Measures semantic preservation of the perturbation.
        Distinguishes harmless orthographic changes (low SES) from meaning‑changing edits (high SES).
        Range: [0, 1].

    Args:
        s: Original text sequence x.
        sp: Perturbed text sequence x~.
        model_name: Optional model identifier.

    Returns:
        float: The semantic entailment distance score.
    """
    model = model_manager.get_model(model_name, model_type="sequence_classification", category="nli")
    tokenizer = model_manager.get_tokenizer(model_name, category="nli")
    device = model_manager.get_device()
    
    def get_entailment_prob(premise, hypothesis):
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            
            # Check model config for entailment label ID
            entailment_idx = 1 # Default assumption
            if hasattr(model.config, 'label2id'):
                entailment_idx = model.config.label2id.get('entailment', 1)
            elif hasattr(model.config, 'id2label'):
                # Reverse search if only id2label exists
                for k, v in model.config.id2label.items():
                    if v.lower() == 'entailment':
                        entailment_idx = k
                        break
            
            return probs[0][entailment_idx].item()

    prob_forward = get_entailment_prob(s, sp)
    prob_backward = get_entailment_prob(sp, s)
    
    return 1.0 - (prob_forward + prob_backward) / 2.0

def embedding_cosine_drift(s: str, sp: str, model_name: str = None) -> float:
    """
    Legacy alias for contextual_embedding_distance.
    """
    return contextual_embedding_distance(s, sp, model_name)
