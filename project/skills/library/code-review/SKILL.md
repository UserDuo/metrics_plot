---
name: code-review
description: Conducts a comprehensive code review focusing on readability, maintainability, performance, and security. Use when asked to review code, PRs, or snippets.
---

# Code Review Workflow

This skill guides the user through a professional code review process. It ensures code quality, security, and maintainability.

## Stage 1: Context & Scope

**Goal:** Understand the code's purpose and the review's constraints.

### Questions
1. What is the intended functionality of this code?
2. Are there specific coding standards (PEP8, Google Style Guide, etc.) to follow?
3. Is this production code, a script, or a prototype?
4. Are there known performance bottlenecks or security concerns?

## Stage 2: Static Analysis (Mental Linting)

**Goal:** Check for syntax, style, and obvious bugs.

### Checklist
- **Style:** Naming conventions, indentation, comments (docstrings).
- **Structure:** Modularization, function length, class design.
- **Safety:** Input validation, error handling (try/except blocks).
- **Complexity:** Cyclomatic complexity (nested loops/ifs).

## Stage 3: Logic & Performance

**Goal:** Verify the algorithm and efficiency.

### Analysis
- **Correctness:** Does the code solve the problem described in Stage 1?
- **Edge Cases:** What happens with empty inputs, huge inputs, or invalid types?
- **Performance:** Are there unnecessary loops, O(N^2) operations where O(N) exists?
- **Resources:** Memory usage, file handle closing, database connection management.

## Stage 4: Feedback & Refactoring

**Goal:** Provide actionable, constructive feedback.

### Output Format
- **Summary:** High-level impression (LGTM, Needs Work).
- **Critical Issues:** Bugs, security flaws (Must Fix).
- **Suggestions:** Performance improvements, style cleanups (Should Fix).
- **Nitpicks:** Typos, variable names (Nice to Fix).
- **Refactored Example:** Provide a snippet showing how to implement the key suggestions.

## Execution Instructions
- Be constructive and polite.
- Focus on the code, not the coder.
- Prioritize correctness and security over style.
