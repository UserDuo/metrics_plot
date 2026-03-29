# A Holistic Metric Framework for Measuring Nuanced Textual Disturbances  
（文本微扰差异的整体度量框架）

## Abstract
This report presents a comprehensive, reproducible, and theoretically grounded framework for measuring nuanced differences between an original sentence and a disturbed sentence. Disturbances include Unicode whitespace variations and orthographic misspellings. Although such perturbations often preserve human-perceived meaning, they can substantially affect information structure, tokenization, semantic representations, rhythm/prosody, and visual rendering.  
We define rich metric families, provide mathematical formulations, explain their linguistic and computational interpretations, specify value ranges, and propose a normalized aggregation strategy suitable for NLP robustness evaluation.

---

## 1. Problem Definition 问题定义

Let:
- **S**: original sentence  
- **S′**: disturbed sentence  
- Disturbance operators:
  - Unicode whitespace substitution or insertion (25 types)
  - Character-level misspellings (insert / delete / substitute)

Goal:  
> Quantify *how* and *where* S′ differs from S, beyond surface string similarity.

---

## 2. Metric Taxonomy 指标体系

| Category | Aspect | Linguistic Level |
|-------|------|-----------------|
| Informatics | Information & distribution | Character / string |
| Tokenization | Subword segmentation | Model interface |
| Semantics | Meaning preservation | Representation |
| Rhythm | Prosody & fluency | Phonological |
| Visualization | Glyph & layout | Visual perception |

Each metric is defined formally and normalized to \[0,1\].

---

## 3. Informatics Metrics 信息论指标

### 3.1 Normalized Levenshtein Distance  
**定义：** 最小编辑距离的长度归一化形式

公式：  
LDₙ(S,S′) = Lev(S,S′) / max(|S|,|S′|)

解释：  
- 衡量最基本的字符修改成本
- 对语义不敏感

取值范围：  
\[0,1\]  
0 = 完全相同；1 = 最大差异

---

### 3.2 Character n-gram Jaccard Distance  
**定义：** 字符 n-gram 集合差异

公式：  
Jₙ = 1 − |Gₙ(S) ∩ Gₙ(S′)| / |Gₙ(S) ∪ Gₙ(S′)|

解释：  
- 捕捉局部拼写结构变化
- 对 misspelling 非常敏感

取值范围：  
\[0,1\]

---

### 3.3 Compression-Based Information Delta  
**定义：** 基于压缩长度的信息复杂度变化

公式：  
ΔC = (C(S′) − C(S)) / C(S)

解释：  
- 近似 Kolmogorov complexity
- whitespace / rare unicode 增大复杂度

取值范围：  
实数（通常接近 0）

---

### 3.4 Shannon Entropy Shift  
**定义：** 字符分布熵变化

公式：  
H(S) = − Σ p(c) log₂ p(c)  
ΔH = H(S′) − H(S)

解释：  
- 衡量字符分布的不可预测性变化

取值范围：  
实数

---

## 4. Tokenization Metrics 分词与碎片化

### 4.1 Token Count Change  
公式：  
ΔT = (|tok(S′)| − |tok(S)|) / |tok(S)|

解释：  
- whitespace 可能导致 token 数骤增
- 直接影响计算成本

范围：  
实数

---

### 4.2 Fragmentation Index  
**定义：** token 序列编辑距离

公式：  
FI = EditDist(tok(S), tok(S′)) / max(|tok(S)|,|tok(S′)|)

解释：  
- 衡量 subword 重分割严重程度
- 对模型鲁棒性极其关键

范围：  
\[0,1\]

---

### 4.3 Token Overlap Ratio  
公式：  
TOR = |tok(S) ∩ tok(S′)| / max(|tok(S)|,|tok(S′)|)

解释：  
- 越低表示 token 漂移越大

范围：  
\[0,1\]

---

## 5. Semantic Metrics 语义指标

### 5.1 Embedding Cosine Drift  
公式：  
Sim = cos(e(S), e(S′))  
SemanticDrift = 1 − (Sim + 1) / 2

解释：  
- 测量语义表示是否稳定
- 对人类等价但模型不稳定情况极其重要

范围：  
\[0,1\]

---

### 5.2 Language Model Surprisal Delta  
公式：  
NLL(X) = −log P(X)  
ΔNLL = (NLL(S′) − NLL(S)) / |tok(S′)|

解释：  
- 模型对扰动的“惊讶程度”
- whitespace unicode 往往异常显著

范围：  
实数（归一化后映射至 \[0,1\]）

---

## 6. Rhythm & Prosody Metrics 节律指标

### 6.1 Syllable Count Change  
公式：  
Δsyll = (syll(S′) − syll(S)) / syll(S)

解释：  
- 拼写错误可能改变音节结构

范围：  
实数

---

### 6.2 Stress Pattern Divergence  
公式：  
SPD = EditDist(σ(S), σ(S′)) / |σ(S)|

解释：  
- 反映朗读节奏变化

范围：  
\[0,1\]

---

## 7. Visualization Metrics 视觉指标

### 7.1 Rendered SSIM Distance  
公式：  
VisualDist = 1 − SSIM(Render(S), Render(S′))

解释：  
- 捕捉不可见但影响布局的 unicode

范围：  
\[0,1\]

---

### 7.2 Glyph Layout Displacement  
公式：  
GLD = (1/n) Σ ||pᵢ − pᵢ′|| / font_size

解释：  
- 衡量排版位移

范围：  
\[0,1\]

---

## 8. Normalization & Aggregation 归一化与综合评分

### 8.1 Metric Normalization  
采用分位数归一化：  
m̂ = (m − P₁) / (P₉₉ − P₁)

---

### 8.2 Composite Disturbance Score  
公式：  
D = wᵢI + wₜT + wₛS + wᵣR + wᵥV  

默认权重：

| Category | Weight |
|-------|--------|
| Informatics | 0.15 |
| Tokenization | 0.25 |
| Semantics | 0.35 |
| Rhythm | 0.10 |
| Visualization | 0.15 |

范围：  
\[0,1\]

---

## 9. Interpretation Guide 结果解释

| D 值 | 含义 |
|----|----|
| <0.05 | 可忽略扰动 |
| 0.05–0.2 | 轻微结构影响 |
| 0.2–0.5 | 显著模型影响 |
| ≥0.5 | 严重语义/解析破坏 |

---

## 10. Applications 应用场景

- LLM 鲁棒性测试
- Unicode 攻击检测
- 数据集质量审计
- Tokenizer 设计评估
- NLP 系统安全分析

---

## 11. Conclusion 结论

This completed report establishes a linguistically motivated, mathematically precise, and operationally reproducible framework for measuring nuanced textual disturbances. It demonstrates that surface-minimal changes can induce multi-layer divergences in modern NLP systems, underscoring the necessity of holistic evaluation.

---

## Appendix
See accompanying code appendix for reproducible HuggingFace-based implementation.
