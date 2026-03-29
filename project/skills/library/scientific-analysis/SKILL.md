---
name: scientific-analysis
description: Guide users through rigorous scientific analysis of text metrics. Use when analyzing results from the BatchStudy, writing research papers, or interpreting metric data. Enforces Nature-standard reporting, statistical rigor, and holistic interpretation. Trigger when user asks to analyze results, interpret data, or write a scientific report.
---

# Scientific Analysis & Reporting Workflow

This skill guides the user through a high-level scientific analysis of text disturbance metrics. It ensures that all findings are reported with statistical rigor and aligned with the "Holistic Metric Framework" design motivation.

## Stage 1: Data Validation & Descriptive Statistics

**Goal:** Ensure data integrity and understand the basic distribution of metrics.

### Step 1: Load and Validate
- Check if `study_results/raw_results.csv` exists.
- Verify that all 6 dimensions (Informatics, Tokenization, Semantics, Syntax, Rhythm, Visualization) are represented.
- Check for missing values or anomalies (e.g., negative distances where impossible).

### Step 2: Descriptive Stats
- Calculate Mean, Median, SD for each metric.
- **Critical:** Do not just list numbers. Interpret the SD. High SD implies the disturbance is context-sensitive; Low SD implies it is systematic/robust.

## Stage 2: Holistic Profile Analysis (The "Fingerprint")

**Goal:** Interpret the Radar Chart (`mean_radar_chart.png`) as a "Disturbance Fingerprint".

### Step 1: Dimensional Breakdown
- **Informatics:** Is the surface change minimal? (High Levenshtein/Jaccard implies obvious errors).
- **Tokenization:** Is the vocabulary collapsing? (High OOV/Byte-fallback).
- **Semantics:** Is the meaning preserved? (High Entailment + Low Cosine Drift).
- **Syntax:** Is the structure stable? (Low POS Divergence).
- **Rhythm:** Is the prosody affected? (Stress patterns).
- **Visualization:** Is the visual glyph shape changed?

### Step 2: Synthesis
- Combine these observations into a narrative.
- *Example:* "The disturbance is 'Stealthy'—it has low Informatics impact (hard to spot) but high Semantic Drift (changes meaning) and high Tokenization impact (breaks LLM processing)."

## Stage 3: Statistical Significance & Correlation

**Goal:** Prove that the observed changes are non-random and understand relationships.

### Step 1: Metric Taxonomy (Clustermap)
- Analyze `metric_taxonomy_clustermap.png`.
- Identify clusters. Do Semantic metrics correlate with Syntax metrics?
- **Hypothesis Generation:** If OOV rate correlates with Semantic Drift, it suggests that vocabulary loss is the mechanism of meaning change.

### Step 2: Significance Testing (if pairs available)
- If comparing multiple disturbance types, suggest T-tests or ANOVA.
- Report p-values. "The increase in POS Divergence is statistically significant (p < 0.05)."

## Stage 4: Nature-Standard Reporting

**Goal:** Draft the Results/Discussion section.

### Reporting Standards
- **Precision:** Use 2-3 decimal places.
- **Uncertainty:** Always report Mean ± SD (or CI).
- **Visuals:** Reference the high-DPI figures generated.
- **Tone:** Objective, cautious, precise. Avoid "huge", "massive"; use "substantial", "statistically significant".

### Structure
1.  **Overview:** "The holistic profile reveals a [Type] disturbance pattern."
2.  **Micro-Analysis:** Detailed breakdown of specific metrics.
3.  **Mechanism:** "The likely mechanism of action is [Mechanism]..."
4.  **Implications:** "This suggests that LLMs relying on [Feature] will be vulnerable."

## Execution Instructions

- When analyzing, always reference the specific files in `study_results/`.
- If the user asks for a report, draft it using the structure above.
- If the user provides new data, run the validation steps first.
