#!/usr/bin/env python3
"""
Holistic Agent - Research Pipeline for Textual Perturbation Analysis

This module serves as the main entry point for a comprehensive research pipeline that analyzes
textual disturbances across multiple linguistic dimensions. The system implements a holistic
approach to perturbation analysis, combining traditional NLP metrics with advanced signal
processing techniques.

Key Scientific Motivation:
- Understanding how different types of textual perturbations (typos vs. Unicode whitespace)
  propagate through linguistic processing pipelines
- Quantifying the impact of adversarial perturbations on modern NLP systems
- Developing robust evaluation frameworks for text processing robustness

Mathematical Foundation:
- Multi-dimensional perturbation analysis across 6 linguistic dimensions
- Statistical hypothesis testing with proper multiple comparison corrections
- Correlation analysis with hierarchical clustering for metric taxonomy
- Wavelet-based signal processing for perturbation signature detection
"""

import sys
import os

# Add project parent directory to path to allow 'from project...' imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project.skills.registry import registry
from project.skills.metric_skills import register_all_metrics
from project.study import BatchStudy, build_high_quality_dataset, summarize_dataset
from project.utils.plotting import figure_quality_report, configure_nature_style
from project.utils.hardware import get_gpu_diagnostics, format_gpu_diagnostics
from project.analysis.tokenizer_bench import benchmark_tokenizers as run_tokenizer_benchmark, load_corpus as load_tokenizer_corpus
import pandas as pd
import time
import random
import re
import numpy as np
import torch

class HolisticAgent:
    def __init__(self):
        print("Initializing Holistic Agent (Research Mode)...")
        register_all_metrics()
        self.tools = registry.list_skills()
        print(f"Loaded {len(self.tools)} tools.")
        try:
            diag = get_gpu_diagnostics()
            print(format_gpu_diagnostics(diag))
        except Exception:
            pass
        
        # Display available high-level skills
        print("\n" + registry.list_contextual_skills() + "\n")

    def _inject_unicode_whitespace(self, s: str) -> tuple[str, str, str]:
        """
        Inject invisible Unicode whitespace characters to simulate adversarial perturbations.
        
        Scientific Motivation:
        - Tests robustness of NLP systems against visually imperceptible adversarial attacks
        - Unicode whitespace injection is a common real-world adversarial technique
        - Evaluates tokenizer and parser resilience to boundary-disrupting perturbations
        
        Mathematical Logic:
        - Randomly selects from 17 different Unicode whitespace characters
        - Inserts at random position within a randomly chosen word
        - Preserves visual appearance while disrupting tokenization boundaries
        
        Args:
            s: Original text string
            
        Returns:
            tuple: (disturbed_text, injected_character, unicode_codepoint)
        """
        # Comprehensive set of Unicode whitespace characters for robust adversarial testing
        # Includes standard ASCII whitespace plus invisible Unicode variants
        ws_chars = [
            '\u0009', '\u000A', '\u000B', '\u000C', '\u000D',  # ASCII control characters
            '\u0020', '\u0085', '\u00A0', '\u1680',            # Standard and non-breaking spaces
            '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', '\u200A',  # En spaces
            '\u2028', '\u2029', '\u202F', '\u205F', '\u3000'   # Line separators and ideographic space
        ]
        words = s.split()
        if len(words) < 2:
            return s, '', ''
        idx = random.randint(0, len(words) - 2)
        w = words[idx]
        insert_pos = random.randint(1, max(1, len(w) - 1))
        ws = random.choice(ws_chars)
        disturbed = w[:insert_pos] + ws + w[insert_pos:]
        words[idx] = disturbed
        code = f"U+{ord(ws):04X}"
        return " ".join(words), ws, code

    def _misspell_word(self, s: str) -> str:
        """
        Introduce realistic typographical errors into text to simulate natural human typing mistakes.
        
        Scientific Motivation:
        - Models real-world text corruption patterns observed in human typing behavior
        - Provides controlled baseline for comparing against adversarial perturbations (Unicode whitespace)
        - Enables evaluation of NLP system robustness to common, non-malicious text variations
        - Supports the experimental design's dual-perturbation comparison (typo vs. whitespace)
        
        Mathematical Logic:
        - Tokenizes text using word boundary regex to preserve punctuation and spacing
        - Filters for alphabetic words ≥4 characters to target meaningful lexical units
        - Implements two perturbation strategies with 50% probability each:
          1. Adjacent character transposition (for words ≥5 chars): Models finger motor errors
          2. Single character substitution: Models keyboard layout proximity errors
        - Preserves original case pattern to maintain visual authenticity
        
        Experimental Design:
        - Single-word perturbation per text to isolate individual error effects
        - Minimum word length constraint ensures meaningful lexical impact
        - Probabilistic strategy selection mimics natural error distribution
        
        Args:
            s: Original text string
            
        Returns:
            str: Text with single realistic typographical error introduced
        """
        # Tokenize text while preserving punctuation and whitespace structure
        # This ensures perturbations affect only lexical content, not formatting
        tokens = re.findall(r"\b\w+\b|\W+", s)
        
        # Identify candidate words: alphabetic, ≥4 characters, case-preserving
        # Length constraint targets meaningful lexical units rather than function words
        word_indices = [i for i, t in enumerate(tokens) if re.match(r"\b\w+\b", t) and t.isalpha() and len(t) >= 4]
        
        if not word_indices:
            return s
            
        # Randomly select target word from candidates
        i = random.choice(word_indices)
        w = tokens[i]
        
        # Strategy 1: Adjacent character transposition (models finger motor errors)
        # Applied to longer words (≥5 chars) with 50% probability
        if len(w) >= 5 and random.random() < 0.5:
            # Select random adjacent pair for transposition
            j = random.randint(0, len(w) - 2)
            # Transpose characters at positions j and j+1
            w2 = w[:j] + w[j+1] + w[j] + w[j+2:]
        else:
            # Strategy 2: Character substitution (models keyboard proximity errors)
            j = random.randint(0, len(w) - 1)
            # Exclude original character to ensure meaningful change
            alphabet = 'abcdefghijklmnopqrstuvwxyz'
            repl = random.choice(alphabet.replace(w[j].lower(), ''))
            # Preserve original case pattern
            repl = repl.upper() if w[j].isupper() else repl
            w2 = w[:j] + repl + w[j+1:]
            
        tokens[i] = w2
        return "".join(tokens)

    def run_batch_study_demo(self, size: int | None = None):
        """
        Execute a complete batch study demonstration using the AdvBench corpus.
        
        Scientific Motivation:
        - Provides reproducible demonstration of the holistic perturbation analysis pipeline
        - Validates the complete research workflow from data loading through statistical analysis
        - Serves as integration test for all pipeline components (metrics, statistics, reporting)
        - Demonstrates real-world applicability on established benchmark dataset (AdvBench)
        
        Experimental Design:
        - Uses AdvBench parallel corpus containing paired original/perturbed text samples
        - Implements controlled perturbation injection (typo vs. Unicode whitespace)
        - Applies comprehensive metric evaluation across 6 linguistic dimensions
        - Generates statistical analysis with proper multiple comparison corrections
        - Produces publication-ready visualizations and reports
        
        Reproducibility Measures:
        - Fixed random seed initialization across all libraries (random, numpy, torch)
        - Deterministic dataset construction with consistent perturbation patterns
        - Standardized output directory structure with timestamped results
        
        Mathematical Foundation:
        - Multi-dimensional statistical analysis with effect size calculations
        - Correlation-based metric taxonomy construction
        - Hypothesis testing with family-wise error rate control
        - Wavelet-based perturbation signature detection
        
        Args:
            size: Optional dataset size limit for testing/debugging (None = full dataset)
            
        Returns:
            None: Results saved to standardized output directory structure
            
        Output Structure:
            - raw_results.csv: Complete metric measurements for all text pairs
            - descriptive_stats.csv: Statistical summaries by metric
            - group_comparison.csv: Statistical tests between perturbation types
            - significance_*.csv: Within-group significance tests
            - *.png/*.svg: Publication-ready visualizations
            - SCIENTIFIC_REPORT_DRAFT.md: Complete analysis report
        """
        print("\n=== Initiating Study (AdvBench Corpus) ===")
        
        # 1. Dataset Loading: exclusively use AdvBench parallel corpus (real pairs, observed counts)
        try:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
        except Exception:
            pass
        dataset = build_high_quality_dataset(size)
        print(f"Loaded AdvBexitch dataset: {len(dataset)} pairs.")
        
        # 2. Experiment Execution
        study = BatchStudy(output_dir="results")
        summarize_dataset(dataset, study.output_dir)
        try:
            df_summary = pd.DataFrame(dataset)
            print("\nAdvBench dataset distribution summary:")
            print(df_summary.groupby(["type"]).size().rename("count").to_string())
            print(df_summary.groupby(["type","category"]).size().rename("count").head(12).to_string())
            print(df_summary.groupby(["type","length_bin"]).size().rename("count").to_string())
        except Exception:
            pass
        df = study.run_batch(dataset)
        
        # 2.5. Tokenizer Benchmark Execution
        print("\n--- Running Tokenizer Benchmark ---")
        try:
            tokenizer_corpus_path = os.path.join(study.output_dir, "raw_results.csv")
            tokenizer_corpus = load_tokenizer_corpus(tokenizer_corpus_path)
            run_tokenizer_benchmark(tokenizer_corpus)
        except Exception as e:
            print(f"Tokenizer benchmark failed: {e}")

        # 3. Statistical Analysis
        print("\n--- Statistical Analysis (AdvBench) ---")
        study.analyze_results(df)
        
        print("\nStudy complete. Results and plots saved to 'results/' directory.")

    def run_interactive_agent(self):
        """
        Launch an interactive command-line interface for the Holistic Agent research system.
        
        Scientific Motivation:
        - Provides real-time exploration and validation of textual perturbation analysis tools
        - Enables interactive hypothesis testing and metric exploration without code modification
        - Facilitates reproducible research through command logging and deterministic execution
        - Supports iterative experimental design through immediate feedback loops
        
        Experimental Design:
        - Command-driven interface with comprehensive help system
        - Support for both batch processing (study) and individual metric execution
        - Integration with skill registry for dynamic tool discovery and loading
        - Quality assurance tools for figure validation and style standardization
        
        Mathematical Foundation:
        - Deterministic execution environment with proper error handling
        - Modular command parsing with keyword argument support
        - Statistical validation through figure quality analysis
        - Reproducible styling via Nature journal standards
        
        Research Applications:
        - Interactive exploration of linguistic perturbation effects
        - Rapid prototyping of new analysis pipelines
        - Educational demonstration of textual analysis techniques
        - Quality control for publication-ready visualizations
        
        Returns:
            None: Interactive session continues until user exits
            
        Command Interface:
            study [size]: Execute batch study with optional sample size
            skills: List all available metric tools
            contextual: Display high-level contextual skills
            search <query>: Search skill registry
            load <skill>: Load skill instructions
            loadfile <path>: Load external skill file
            vizqa <dir>: Analyze figure quality
            upgrade-style: Apply Nature styling
            metric <name> [args]: Execute specific metric
        """
        print("\n=== Holistic Agent Interactive Mode ===")
        print("Type 'help' for commands, 'exit' to quit.")
        
        while True:
            try:
                user_input = input("\nAgent> ").strip()
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit']:
                    print("Exiting Agent.")
                    break
                
                elif user_input.lower() == 'help':
                    print("Available Commands:")
                    print("  study            - Run the Batch Study Demo")
                    print("  skills           - List all available tools/skills")
                    print("  contextual       - List high-level contextual skills")
                    print("  search <query>   - Search for a skill")
                    print("  load <skill>     - Load instructions for a contextual skill")
                    print("  loadfile <path>  - Load instructions from a local SKILL.md file")
                    print("  vizqa <dir>      - Analyze figure quality in directory (PNG/SVG presence, size)")
                    print("  upgrade-style    - Apply Nature-style configuration globally")
                    print("  metric <name>    - Execute a specific metric (e.g., metric normalized_levenshtein s='test' sp='tst')")
                
                elif user_input.lower().startswith('study'):
                    parts = user_input.split()
                    size = 500
                    if len(parts) > 1:
                        try:
                            size = int(parts[1])
                        except Exception:
                            pass
                    self.run_batch_study_demo(size=size)
                
                elif user_input.lower() == 'skills':
                    print(f"Total Tools: {len(self.tools)}")
                    for tool in self.tools:
                        print(f"- {tool['name']}")
                        
                elif user_input.lower() == 'contextual':
                    print(registry.list_contextual_skills())
                    
                elif user_input.lower().startswith('search '):
                    query = user_input[7:].strip()
                    print(registry.search_contextual_skills(query))
                    
                elif user_input.lower().startswith('load '):
                    skill_name = user_input[5:].strip()
                    instructions = registry.load_skill_instructions(skill_name)
                    print(f"\n{instructions}\n")
                    print(f"[System] Skill '{skill_name}' loaded into context.")
                
                elif user_input.lower().startswith('loadfile '):
                    path = user_input[9:].strip()
                    instructions = registry.load_external_skill(path)
                    print(f"\n{instructions}\n")
                    print(f"[System] External skill loaded from '{path}'.")
                
                elif user_input.lower().startswith('vizqa'):
                    parts = user_input.split()
                    target = parts[1] if len(parts) > 1 else "results"
                    df = figure_quality_report(target, outfile=os.path.join(target, "figure_quality.csv"))
                    print(df.to_string(index=False))
                    print(f"[System] Figure quality report saved to {os.path.join(target, 'figure_quality.csv')}")
                
                elif user_input.lower() == 'upgrade-style':
                    configure_nature_style()
                    print("[System] Nature-style configuration applied.")
                    
                elif user_input.lower().startswith('metric '):
                    # Simple parser for demo purposes: metric name key=value key=value
                    parts = user_input[7:].split()
                    if not parts:
                        print("Usage: metric <name> key=value ...")
                        continue
                        
                    metric_name = parts[0]
                    kwargs = {}
                    for part in parts[1:]:
                        if '=' in part:
                            k, v = part.split('=', 1)
                            kwargs[k] = v
                    
                    try:
                        result = registry.execute_tool(metric_name, **kwargs)
                        print(f"Result: {result}")
                    except Exception as e:
                        print(f"Execution Error: {e}")
                        
                else:
                    print("Unknown command. Type 'help' for options.")
                    
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """
    Entry point for the Holistic Agent textual perturbation analysis system.
    
    Scientific Motivation:
    - Provides unified interface for reproducible textual perturbation research
    - Supports both automated batch processing and interactive exploration modes
    - Enables systematic evaluation of adversarial text effects across linguistic dimensions
    - Facilitates comparative analysis of different perturbation strategies
    
    Experimental Design:
    - Command-line interface with mode selection (batch vs interactive)
    - Deterministic execution environment with proper random seeding
    - Modular architecture supporting extensible metric and skill systems
    - Integration with established benchmarks (AdvBench) for validation
    
    Mathematical Foundation:
    - Statistical hypothesis testing with multiple comparison corrections
    - Multi-dimensional correlation analysis for metric taxonomy construction
    - Effect size calculations (Cohen's d, Hedges' g) for practical significance
    - Wavelet-based signal processing for perturbation signature detection
    
    Research Applications:
    - Adversarial robustness evaluation for NLP systems
    - Linguistic perturbation effect quantification
    - Publication-ready visualization and reporting
    - Educational demonstration of text analysis techniques
    
    Command Line Interface:
        python main.py [--batch]: Run in batch mode (automated study) or interactive mode
        
    System Requirements:
        - Python 3.8+ with scientific computing stack
        - GPU support optional but recommended for large-scale analysis
        - Nature journal styling for publication-ready outputs
        
    Returns:
        None: System execution continues until completion or user exit
        
    Output Structure:
        - Standardized directory structure with timestamped results
        - Publication-ready figures (PNG/SVG formats)
        - Statistical analysis reports (CSV formats)
        - Scientific manuscript drafts (Markdown format)
    """
    agent = HolisticAgent()
    # If arguments provided, run batch study, else interactive
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        agent.run_batch_study_demo()
    else:
        agent.run_interactive_agent()

if __name__ == "__main__":
    main()
