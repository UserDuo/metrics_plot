"""
Tokenizer Robustness Benchmark (Renyi Entropy).

This module evaluates how sensitive different tokenization algorithms are to real-world text
perturbations. Given a parallel corpus of (original, perturbed) texts, it tokenizes each side
with multiple Hugging Face tokenizers and computes a distributional stability score using
Renyi entropy (power=2.5) over token sequences.

Interpretation:
- Smaller |delta| suggests that the tokenizer preserves a similar token distribution under noise.
- Larger shifts can indicate boundary instability (e.g., fragmentation into rare subpieces).

Outputs:
- Prints a compact console table.
- Writes `results/tokenizer_benchmark.csv` for inclusion in reports.
"""

import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer
import tokenization_scorer
from typing import List, Dict
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

def _renyi_entropy(tokens: list[str], alpha: float = 2.5) -> float:
    from collections import Counter
    if not tokens:
        return 0.0
    cnt = Counter(tokens)
    vals = np.array(list(cnt.values()), dtype=float)
    p = vals / np.sum(vals)
    if alpha == 1.0:
        return float(-np.sum(p * np.log(p + 1e-12)))
    return float((1.0 / (1.0 - alpha)) * np.log(np.sum(p ** alpha) + 1e-12))

def _detect_lang(text: str) -> str:
    import re
    if not text:
        return "unknown"
    ascii_letters = len(re.findall(r"[A-Za-z]", text))
    total_letters = len(re.findall(r"\w", text))
    ratio = (ascii_letters / max(1, total_letters))
    return "en" if ratio >= 0.7 else "other"

def load_corpus(csv_path: str) -> List[tuple]:
    """
    Load a parallel corpus from `raw_results.csv`.

    Args:
        csv_path: Path to a CSV containing `original_text` and `perturbed_text` columns.

    Returns:
        A list of (original_text, perturbed_text) pairs as strings.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Corpus file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    # Filter for valid strings
    df = df.dropna(subset=['original_text', 'perturbed_text'])
    
    corpus = []
    for _, row in df.iterrows():
        corpus.append((str(row['original_text']), str(row['perturbed_text'])))
    return corpus

def benchmark_tokenizers(corpus: List[tuple]):
    """
    Benchmark multiple tokenizer algorithms on a parallel corpus using Renyi entropy.

    The benchmark measures the change in Renyi entropy between token sequences produced from
    original vs. perturbed texts. The goal is not to compare absolute entropy across models,
    but to compare relative stability under perturbation.

    Args:
        corpus: List of (original_text, perturbed_text) pairs.

    Returns:
        A list of result dictionaries; also persists a CSV under `results/tokenizer_benchmark.csv`.
    """
    
    # Define tokenizer candidates covering 4 main types
    # 1. WordPiece (BERT)
    # 2. BBPE (GPT-2)
    # 3. Unigram (ALBERT - via SentencePiece)
    # 4. BPE (OpenAI-GPT - Standard BPE)
    
    candidates = [
        {"name": "bert-base-uncased", "type": "WordPiece"},
        {"name": "gpt2", "type": "BBPE"},
        {"name": "t5-base", "type": "Unigram"}
    ]
    
    print(f"{'Model':<30} | {'Type':<10} | {'Metric':<10} | {'Original':<10} | {'Perturbed':<10} | {'Delta':<10}")
    print("-" * 95)
    
    results = []
    details_rows = []
    lengths = [len(orig) for orig, _ in corpus]
    if lengths:
        q = np.quantile(lengths, [0.25, 0.5, 0.75]).tolist()
    else:
        q = [0, 0, 0]

    for cand in candidates:
        model_name = cand["name"]
        algo_type = cand["type"]
        
        try:
            # Instantiate tokenizer
            hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Prepare tokenized lists
            originals_tokens = []
            perturbeds_tokens = []
            
            # Batch tokenize for efficiency? Simple loop is fine for small corpus
            for orig, pert in corpus:
                # tokenize() returns list of strings
                originals_tokens.append(hf_tokenizer.tokenize(orig))
                perturbeds_tokens.append(hf_tokenizer.tokenize(pert))
            
            # Sample-level Renyi entropy (H_alpha) per text
            sample_orig = []
            sample_pert = []
            for i, ((orig_text, pert_text), to, tp) in enumerate(zip(corpus, originals_tokens, perturbeds_tokens)):
                Ho = _renyi_entropy(to, alpha=2.5)
                Hp = _renyi_entropy(tp, alpha=2.5)
                sample_orig.append(Ho)
                sample_pert.append(Hp)
                ln = len(orig_text)
                if ln <= q[0]:
                    lq = "Q1"
                elif ln <= q[1]:
                    lq = "Q2"
                elif ln <= q[2]:
                    lq = "Q3"
                else:
                    lq = "Q4"
                lang = _detect_lang(orig_text)
                details_rows.append({
                    "model": model_name,
                    "type": algo_type,
                    "sample_id": i,
                    "length_chars": ln,
                    "length_quantile": lq,
                    "language": lang,
                    "original_score": Ho,
                    "perturbed_score": Hp,
                    "delta": Hp - Ho
                })
            
            # Corpus-level means for summary
            score_orig = float(np.mean(sample_orig)) if sample_orig else 0.0
            score_pert = float(np.mean(sample_pert)) if sample_pert else 0.0
            delta = float(score_pert - score_orig)
            
            print(f"{model_name:<30} | {algo_type:<10} | {'Renyi':<10} | {score_orig:.4f}     | {score_pert:.4f}      | {delta:+.4f}")
            
            results.append({
                "model": model_name,
                "type": algo_type,
                "metric": "Renyi (p=2.5)",
                "original_score": score_orig,
                "perturbed_score": score_pert,
                "delta": delta
            })
            
        except Exception as e:
            print(f"{model_name:<30} | {algo_type:<10} | Error: {str(e)}")

    # Save results to CSV
    if results:
        res_df = pd.DataFrame(results)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_path = os.path.join(base_dir, "results", "tokenizer_benchmark.csv")
        res_df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")

        # Save detailed sample-level CSV
        if details_rows:
            det_df = pd.DataFrame(details_rows)
            details_path = os.path.join(base_dir, "results", "tokenizer_benchmark_details.csv")
            det_df.to_csv(details_path, index=False)
            print(f"Details saved to {details_path}")

        # Generate plot
        try:
            from project.utils.plotting import plot_tokenizer_comparison
            plot_path = output_path.replace(".csv", ".png")
            plot_tokenizer_comparison(output_path, plot_path)
        except ImportError:
            print("Could not import plotting utilities.")
        except Exception as e:
            print(f"Plot generation failed: {e}")

    return results

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path = os.path.join(base_dir, "results", "raw_results.csv")
    
    print(f"Loading corpus from: {csv_path}")
    try:
        corpus = load_corpus(csv_path)
        print(f"Loaded {len(corpus)} pairs.")
        
        benchmark_tokenizers(corpus)
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
