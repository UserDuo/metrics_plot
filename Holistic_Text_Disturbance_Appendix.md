# Appendix: Reproducible Code and Experimental Evaluation

## Appendix A: Reproducible Code (Python, HuggingFace-Based)

### A.1 Environment and Dependencies

```bash
python>=3.9
torch>=2.0
transformers>=4.40
tokenizers>=0.15
sentencepiece
scikit-learn
numpy
scipy
pillow
opencv-python
python-Levenshtein
```

Optional (prosody):
```bash
cmudict
g2p-en
```

---

### A.2 Model and Tokenizer Setup

```python
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embed_model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
).to(DEVICE)
embed_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

lm_model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
lm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

---

### A.3 Metric Implementations

#### Character-Level Metrics

```python
import Levenshtein, gzip, math
from collections import Counter

def normalized_levenshtein(s, sp):
    return Levenshtein.distance(s, sp) / max(len(s), len(sp))

def compression_delta(s, sp):
    return (len(gzip.compress(sp.encode())) - 
            len(gzip.compress(s.encode()))) / len(gzip.compress(s.encode()))

def shannon_entropy(s):
    cnt = Counter(s)
    return -sum((v/len(s))*math.log2(v/len(s)) for v in cnt.values())
```

#### Tokenization Metrics

```python
def tokens(x):
    return tokenizer.tokenize(x)

def fragmentation_index(s, sp):
    return Levenshtein.distance(
        " ".join(tokens(s)), " ".join(tokens(sp))
    ) / max(len(tokens(s)), len(tokens(sp)))
```

#### Semantic Metrics

```python
import torch.nn.functional as F

def sentence_embedding(x):
    inp = embed_tokenizer(x, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = embed_model(**inp)
    return out.last_hidden_state.mean(dim=1).squeeze()

def semantic_drift(s, sp):
    v1, v2 = sentence_embedding(s), sentence_embedding(sp)
    return 1 - (F.cosine_similarity(v1, v2, dim=0).item() + 1) / 2
```

#### Language Model Surprisal

```python
def normalized_nll(x):
    enc = lm_tokenizer(x, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        loss = lm_model(**enc, labels=enc["input_ids"]).loss
    return loss.item()
```

---

## Appendix B: Formal Experimental Design

### B.1 Objectives

- Quantify effects of Unicode whitespace and misspellings.
- Measure divergence across tokenization, semantics, and visualization.
- Identify predictors of downstream instability.

---

### B.2 Dataset Construction

- Base corpus: Wikipedia + news sentences
- Length: 5–30 tokens
- Perturbations:
  - Unicode whitespace (U+0020, U+00A0, U+200B, U+2003)
  - Misspellings (delete, substitute, keyboard-adjacent)

---

### B.3 Experimental Factors

| Factor | Levels |
|------|-------|
| Tokenizer | BPE, WordPiece |
| Model | Small / Base |
| Perturbation | Whitespace / Misspelling |

---

### B.4 Evaluation Metrics

- Composite disturbance score
- Token fragmentation index
- Semantic drift
- ΔNLL

---

### B.5 Statistical Analysis

- Paired bootstrap (10k resamples)
- Effect size (Cohen's d)
- Spearman correlation across metrics

---

### B.6 Hypotheses

H1: Whitespace causes higher token fragmentation than character distance suggests.  
H2: Misspellings induce higher semantic drift.  
H3: Token fragmentation predicts task degradation better than edit distance.

---

### B.7 Reproducibility Checklist

- Fixed seeds
- Logged tokenizer/model versions
- Explicit Unicode codepoints
- Deterministic rendering
