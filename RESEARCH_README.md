# Research: Holistic Textual Disturbance Framework

**Current Status:** Pilot Study (v0.2)
**Maintainers:** Holistic Agent Team
**Associated Codebase:** `project/`

---

## 1. Research Context

### Problem Statement
Large Language Models (LLMs) are increasingly deployed in safety-critical environments. However, they remain vulnerable to "nuanced textual disturbances"—subtle alterations (e.g., typos, whitespace injection, synonym substitution) that humans easily ignore but that can cause model failure (hallucination, refusal, misinterpretation).

### Goals
1.  **Develop a Holistic Metric Framework:** Move beyond simple Levenshtein distance. We need to measure disturbance across multiple dimensions:
    *   **Informatics:** Surface-level noise.
    *   **Tokenization:** Vocabulary collapse and segmentation fragmentation.
    *   **Semantics:** Meaning preservation vs. drift.
    *   **Syntax:** Structural stability (POS tags, dependency trees).
    *   **Rhythm:** Prosodic features (stress patterns).
    *   **Visualization:** Visual glyph similarity.
2.  **Quantify "Stealthiness":** A dangerous attack is one that changes the model's behavior (High Semantic/Tokenization impact) while looking innocent to humans (Low Informatics/Visual impact).
3.  **Standardize Reporting:** Establish a rigorous protocol for reporting these disturbances, compatible with top-tier scientific journals (e.g., *Nature*).

---

## 2. Methodology

### The Holistic Agent
We utilize an agentic framework to orchestrate the research:
-   **Skill-Based Architecture:** The Agent dynamically loads "Contextual Skills" (e.g., `scientific-analysis`, `code-review`) to guide its actions.
-   **Operator Library:** Metrics are encapsulated as independent, reusable operators in `project/metrics/`.

### Metric Dimensions
| Dimension | Key Metrics | Interpretation |
| :--- | :--- | :--- |
| **Informatics** | Normalized Levenshtein, Char Jaccard | How much did the text change on the surface? |
| **Tokenization** | OOV Rate Change, Fragmentation Index | Did the tokenizer break the words differently? |
| **Semantics** | Embedding Cosine Drift, Entailment Score | Did the meaning change? Is it logically consistent? |
| **Syntax** | POS Divergence, Tree Depth Change | Did the grammatical structure shift? |
| **Rhythm** | Stress Pattern Divergence | Does it sound different when read aloud? |
| **Visualization** | SSIM Distance | Does it look different visually? |

---

## 4. Recent Findings: Tokenizer Robustness (January 2026)

A recent benchmark study was conducted to evaluate the robustness of three mainstream tokenization strategies against adversarial text perturbations. The full report can be found in the following formats:
- [Markdown Report](results/tokenizer_benchmark_report.md)
- [DOCX Report](results/tokenizer_benchmark_report.docx)
- [PPTX Presentation](results/tokenizer_benchmark_report.pptx)

### Key Takeaways:
- **Unigram (`albert-base-v2`)** demonstrated the highest stability, making it a recommended choice for safety-critical applications.
- **Byte-level BPE (`gpt2`)** showed significant sensitivity to perturbations, suggesting its potential as a sensor for detecting adversarial attacks.
- **WordPiece (`bert-base-uncased`)** exhibited a moderate increase in entropy, indicating a tendency to create new sub-words from misspellings.

To reproduce the tokenizer benchmark, run:
```bash
python project/analysis/tokenizer_bench.py
```

---

## 5. Usage & Reproduction

### Prerequisites
-   Python 3.8+
-   Dependencies: `requirements.txt`
-   API Keys (Optional): For OpenAI/Anthropic models if configured (default uses local HuggingFace models).

### Running a Study
To execute the standard batch study protocol:

```bash
python project/main.py --batch
```

This will:
1.  Load the simulated dataset.
2.  Execute all 20+ metrics on each pair.
3.  Generate `study_results/`:
    *   `raw_results.csv`: Raw metric scores.
    *   `descriptive_stats.csv`: Mean/SD for each metric.
    *   `mean_radar_chart.png`: The "Holistic Disturbance Profile".
    *   `SCIENTIFIC_REPORT_DRAFT.md`: An auto-generated analysis draft.

### Interactive Exploration
To explore specific metrics or skills:

```bash
python project/main.py
```

Inside the shell:
-   `skills`: List all tools.
-   `metric <name> s="text1" sp="text2"`: Test a specific operator.

---

## 4. Contributing
*   **New Metrics:** Add functions to `project/metrics/` and register them in `project/skills/metric_skills.py`.
*   **New Skills:** Add `SKILL.md` to `project/skills/library/`.
*   **Code Review:** All PRs must pass the `code-review` skill checklist.

---
*This document was co-authored by the Holistic Agent using the `doc-coauthoring` skill.*
