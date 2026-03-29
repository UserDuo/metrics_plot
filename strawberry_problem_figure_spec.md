# Figure Specification  
**Title:** The strawberry problem

## 1. Figure Intent (Scientific Message)
This figure explains the *strawberry problem* as a structural limitation of token-based probabilistic language models, rather than a superficial counting error. It emphasizes that failures at the character level arise from the interaction between tokenization, continuous representations, and training objectives.

**Core causal message:**  
Text → Tokenization (subword/byte) → Continuous representations → Probabilistic output → Approximate character reasoning

---

## 2. Layout and Aspect Ratio (Nature-compliant)
- Aspect ratio: **1.4 : 1**
- Canvas size (vector): **180 mm × 128 mm**
- Margins: **≥ 8 mm** white margin on all sides
- Background: pure white (#FFFFFF)

---

## 3. Visual Structure (Left-to-Right Flow)

### Panel A — Input Text
- Display the word **“strawberry”** as plain text.
- Sans-serif font (Helvetica / Arial / Source Sans Pro)
- Regular weight, dark gray (#333333)

*Annotation:* Raw text input

---

### Panel B — Tokenization
Show two parallel tokenization examples:

**B1. Subword tokenization**
- Tokens: `st` `raw` `berry`
- Rounded rectangles, muted neutral tones
- Label: *Subword tokenization (BPE / unigram)*

**B2. Byte-level tokenization**
- Tokens: `s` `t` `r` `aw` `ber` `ry` (illustrative)
- Smaller rounded rectangles
- Label: *Byte-level tokenization*

*Side note:* Token boundaries depend on tokenizer design and vocabulary.

---

### Panel C — Representation Space
Three stacked modules:
1. Embedding lookup  
2. Contextual encoder (self-attention)  
3. Output distribution (softmax)

- Thin rectangular outlines, light gray stroke
- Faded point clouds to suggest continuous vector space
- No characters or counters shown

---

### Panel D — Output Behavior
- Probability bars instead of binary correctness:
  - P(2 r’s) — longer bar
  - P(3 r’s) — shorter bar
- Muted grayscale or soft blue tones

Label: *Probabilistic prediction*

---

## 4. Mechanism Callout (Inset)
**Sources of failure (non-exhaustive):**
- Subword merging obscures character boundaries
- Training objective optimizes next-token prediction
- No explicit character-count supervision
- Representations favor semantic coherence over symbol precision

---

## 5. Explicit Exclusions
- No character tiles implying token = character
- No red/green binary correctness markers
- No glowing or anthropomorphic model icons
- No deterministic counters or exact arithmetic
- No decorative gradients or shadows

---

## 6. Color Palette
- Text: #2B2B2B  
- Token blocks: #EDEAE4 / #F2F2F2  
- Arrows: #8A8A8A  
- Probability bars: #6F8FAF  

---

## 7. Line and Typography
- Line width: 0.5–0.75 pt
- Arrowheads: small, open
- Typeface: Helvetica / Arial / Source Sans Pro
- Bold text only for the figure title

---

