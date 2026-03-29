# Skills Framework Documentation

This project integrates a robust **Skills Framework** inspired by Anthropic's OpenSkills and "Awesome-Claude-Skills". It allows the Holistic Agent to dynamically load high-level contextual skills and execute precise metric operators.

## Architecture

The framework consists of three main components:

1.  **Registry (`project/skills/registry.py`)**: The central hub that manages all available tools and skills. It supports:
    *   **Metric Tools**: Python functions (operators) for calculating disturbance metrics.
    *   **Contextual Skills**: High-level workflow guides (Markdown instructions) for complex tasks.
2.  **Loader (`project/skills/loader.py`)**: A dynamic loader that scans the `project/skills/library` directory for skill definitions (`SKILL.md` files).
3.  **Library (`project/skills/library/`)**: A collection of skill definitions.

## Available Skills

### Contextual Skills (Workflows)

These skills provide the Agent with detailed instructions ("System Prompts") on how to handle complex scenarios.

*   **`scientific-analysis`**: Guides the rigorous analysis of batch study results, ensuring Nature-standard reporting and holistic profile interpretation.
*   **`doc-coauthoring`**: A structured workflow for collaborating on documentation (Context Gathering -> Refinement -> Testing).
*   **`code-review`**: A checklist-based workflow for conducting professional code reviews (Style, Logic, Performance, Security).
*   **`architecture-design`**: A step-by-step guide for designing software systems (Requirements -> Patterns -> Components).

### Metric Tools (Operators)

These are executable Python functions registered from the `project/metrics/` modules.

*   `normalized_levenshtein`, `char_ngram_jaccard`, `semantic_entailment_score`, `pos_divergence`, etc.

## Usage

### Interactive Mode

Run the agent in interactive mode to explore skills:

```bash
python project/main.py
```

**Commands:**
*   `help`: Show available commands.
*   `contextual`: List available high-level skills.
*   `search <query>`: Search for a skill (e.g., `search code`).
*   `load <skill>`: Load the instructions for a skill (e.g., `load scientific-analysis`).
*   `metric <name> key=value`: Execute a metric (e.g., `metric normalized_levenshtein s='test' sp='tst'`).
*   `study`: Run the Batch Study Demo.

### Batch Mode

Run the pilot study directly:

```bash
python project/main.py --batch
```

## Adding New Skills

1.  Create a new directory in `project/skills/library/<skill-name>`.
2.  Add a `SKILL.md` file with the following structure:

```markdown
---
name: skill-name
description: A brief description of what this skill does.
---

# Skill Title

## Instructions
Detailed step-by-step instructions for the Agent...
```

The `SkillLoader` will automatically discover the new skill upon restart.
