#!/usr/bin/env python3
"""
Textual Perturbation Analysis Engine - Batch Study Module

Scientific Motivation:
This module implements the core batch processing engine for evaluating textual perturbation effects
across multiple linguistic dimensions. The system is designed to support adversarial robustness
research by providing quantitative metrics for character-level, token-level, and semantic-level
disturbances in natural language processing systems.

Experimental Design:
- Batch processing of paired text samples (original vs. perturbed)
- Multi-dimensional metric evaluation across 6 linguistic categories
- Statistical hypothesis testing with multiple comparison corrections
- Reproducible analysis pipeline with deterministic random seeding
- Integration with established benchmarks (AdvBench) for validation

Mathematical Foundation:
- Descriptive statistics with confidence intervals
- Effect size calculations (Cohen's d, Hedges' g, Cliff's delta)
- Correlation analysis with hierarchical clustering
- Wavelet-based signal processing for perturbation signature detection
- Non-parametric statistical tests for robust inference

Research Applications:
- Adversarial text detection and classification
- NLP system robustness evaluation
- Linguistic perturbation effect quantification
- Publication-ready statistical reporting
- Educational demonstration of text analysis techniques

Architecture Overview:
The BatchStudy class serves as the primary orchestrator for textual perturbation analysis,
providing methods for batch processing, statistical analysis, and report generation. The system
integrates with a skill registry for extensible metric execution and supports both automated
batch processing and interactive exploration modes.
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from typing import List, Dict
from tqdm import tqdm
import warnings
import importlib
import importlib.util
from project.skills.registry import SkillRegistry
from project.utils.config import config
from project.skills.registry import registry
from project.skills import metric_skills
from project.utils.plotting import plot_holistic_radar
from project.analysis.reporting import ReportGenerator
from scipy import stats as sstats
from project.utils.plotting import configure_nature_style
from project.utils.plotting import generate_qc_report, figure_quality_report
import re
import shutil
from project.config import FIGURES_DIR_NAME, TABLES_DIR_NAME, MIN_FONT, SHAP_MAX_DISPLAY_BAR, SHAP_MAX_DISPLAY_SWARM

class BatchStudy:
    """
    Orchestrate the batch evaluation and analysis workflow for textual perturbation experiments.

    This class is responsible for:
    - Running the registered metric suite on (original, perturbed) text pairs
    - Persisting raw metric outputs as a flat table (CSV)
    - Performing univariate and multivariate analyses (descriptives, correlations, group tests)
    - Generating publication-oriented artifacts (figures, tables, and report drafts)

    Data Contract:
    - Input dataset items are dictionaries with at least:
      - original: str (clean reference text)
      - disturbed: str (perturbed text)
      - type: str (e.g., "Typo" or "Whitespace")
    - Additional optional fields (e.g., category, length_bin, ws_char/ws_code) are carried through
      to facilitate stratified analysis and auditability.
    """
    def __init__(self, output_dir: str = "results"):
        """
        Initialize the study output location and reporting backend.

        Args:
            output_dir: Directory used to store all artifacts for a run. The directory is created
                if it does not exist.
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.reporter = ReportGenerator(output_dir)

    def run_batch(self, dataset: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Execute the metric suite for each (original, perturbed) pair in the dataset.

        Experimental Role:
        This method implements the measurement stage of the pipeline: for each sample pair,
        it computes a consistent set of scalar metrics designed to capture perturbation impact
        across multiple linguistic dimensions (surface form, tokenization, semantics, syntax,
        rhythm, and rendering/visual layout).

        Robustness Guarantees:
        - Per-metric failures are isolated: individual metric errors are recorded as None while
          keeping the remaining metrics for the sample.
        - Wavelet features are optional: if the wavelet backend or its dependencies are unavailable,
          the study still completes without wavelet columns.

        Args:
            dataset: List of dictionaries. Expected keys:
                - original: original text (str)
                - disturbed: perturbed text (str)
                - type: perturbation type label (str)

        Returns:
            A pandas DataFrame where each row corresponds to one input pair, containing metadata
            columns (original_text, perturbed_text, type, ...) and metric output columns.
        """
        results = []
        
        # Get all registered metric tools
        # We filter for tools that are likely metrics (heuristics or explicit list)
        # For now, we manually define the list of metrics we care about to avoid running management tools
        metric_tools = [
            "normalized_levenshtein", "char_ngram_jaccard", "compression_delta",
            "token_count_change", "fragmentation_index",
            "lm_surprisal_delta", "semantic_entailment_score",
            "syllable_count_change", "stress_pattern_divergence",
            "ssim_distance", "glyph_displacement",
            "dependency_overlap_score", "tree_depth_change", "pos_divergence"
        ]
        metric_tools += [
            "normalized_entropy_delta",
            "contextual_embedding_distance",
            "spatial_dispersion_salience_score",
        ]
        
        # Initialize Wavelet Analyzer
        try:
            from project.utils.wavelet import TextWaveletAnalyzer
            wavelet_analyzer = TextWaveletAnalyzer()
        except ImportError:
            wavelet_analyzer = None
            
        top_wavelet_score = -1
        top_wavelet_pair = None
        top_wavelet_idx = -1
        
        print(f"Starting batch study on {len(dataset)} pairs...")
        
        for i, item in enumerate(tqdm(dataset)):
            s = item['original']
            sp = item['disturbed']
            row = {
                    "original_text": s,
                    "perturbed_text": sp,
                    "type": item.get("type", "Unknown"),
                    "ws_char": item.get("ws_char", None),
                    "ws_code": item.get("ws_code", None),
                    "category": item.get("category", "General"),
                    "length_bin": item.get("length_bin", "medium")
                }
            
            for tool_name in metric_tools:
                try:
                    # Execute via registry
                    val = registry.execute_tool(tool_name, s=s, sp=sp)
                    row[tool_name] = val
                except Exception as e:
                    # print(f"Error running {tool_name}: {e}")
                    row[tool_name] = None
            
            # Compute Wavelet Energy Features
            if wavelet_analyzer:
                try:
                    wf = wavelet_analyzer.compute_wavelet_features(s, sp)
                    row.update(wf)
                    
                    # Track top perturbation for Micro-Scalogram visualization
                    if wf.get('wavelet_energy', 0) > top_wavelet_score:
                        top_wavelet_score = wf['wavelet_energy']
                        top_wavelet_pair = (s, sp)
                        top_wavelet_idx = i
                        
                    # Remove non-scalar cwt_matrix from row to keep DataFrame clean
                    if 'cwt_matrix' in row:
                        del row['cwt_matrix']
                except Exception:
                    pass

            results.append(row)
            
        # Save top scalogram if exists
        if top_wavelet_pair is not None:
            try:
                from project.utils.plotting import plot_perturbation_pipeline
                s_top, sp_top = top_wavelet_pair
                
                # Full Pipeline Dashboard (ABCD)
                plot_perturbation_pipeline(
                    s_top,
                    sp_top,
                    filename=os.path.join(self.output_dir, "perturbation_pipeline.png")
                )
                print(f"Saved perturbation pipeline dashboard to perturbation_pipeline.png")

                try:
                    from project.utils.wavelet import compute_tokenizer_embedding_cwt_delta
                    from project.utils.plotting import plot_tokenizer_cwt_triptych
                    mats = {}
                    mats["WordPiece"] = compute_tokenizer_embedding_cwt_delta(s_top, sp_top, "bert-base-uncased")["delta_energy"]
                    mats["byte-level BPE"] = compute_tokenizer_embedding_cwt_delta(s_top, sp_top, "gpt2")["delta_energy"]
                    mats["Unigram"] = compute_tokenizer_embedding_cwt_delta(s_top, sp_top, "t5-base")["delta_energy"]
                    plot_tokenizer_cwt_triptych(
                        mats,
                        filename=os.path.join(self.output_dir, "tokenizer_embedding_cwt_pipeline.png"),
                    )
                except Exception as e:
                    print(f"Failed to save tokenizer embedding CWT triptych: {e}")
                
            except Exception as e:
                print(f"Failed to save detailed wavelet dynamics: {e}")

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.output_dir, "raw_results.csv"), index=False)
        return df

    def analyze_results(self, df: pd.DataFrame):
        """
        Analyze metric outputs and generate plots/tables/reports.

        Scientific Outputs:
        - Metric coverage audit (presence and non-null counts)
        - Descriptive statistics table for all usable numeric metrics
        - Correlation structure used to derive a metric taxonomy (clustermap)
        - Group comparison (when multiple perturbation types exist) with effect sizes and CIs
        - Multiple-testing control (FDR q-values)
        - Optional discriminative modeling (when dependencies are available) and explanation artifacts
        - Report drafts in Markdown/PPTX/DOCX via the reporting backend

        Statistical Notes:
        - Group comparison uses Welch's t-test when variance is non-degenerate; otherwise falls back
          to non-parametric Mann–Whitney U testing.
        - Practical significance is reported via effect sizes (Cohen's d and Hedges' g) and
          ordinal effect (Cliff's δ).

        Args:
            df: Output table produced by run_batch().
        """
        eps = 1e-12
        expected_metrics = [
            "normalized_levenshtein", "char_ngram_jaccard", "compression_delta",
            "token_count_change", "fragmentation_index",
            "lm_surprisal_delta", "semantic_entailment_score",
            "syllable_count_change", "stress_pattern_divergence",
            "ssim_distance", "glyph_displacement",
            "dependency_overlap_score", "tree_depth_change", "pos_divergence",
            "normalized_entropy_delta", "contextual_embedding_distance", "spatial_dispersion_salience_score"
        ]
        for m in expected_metrics:
            if m in df.columns:
                df[m] = pd.to_numeric(df[m], errors='coerce')
        numeric_cols = [m for m in expected_metrics if m in df.columns]
        coverage_rows = []
        for m in expected_metrics:
            present = m in df.columns
            if present:
                col = df[m]
                non_null = int(col.notna().sum())
                dtype_str = str(col.dtype)
                coverage_ok = non_null >= 3
            else:
                non_null = 0
                dtype_str = ""
                coverage_ok = False
            coverage_rows.append({"metric": m, "present": present, "non_null_count": non_null, "dtype": dtype_str, "coverage_ok": coverage_ok})
        coverage_df = pd.DataFrame(coverage_rows)
        coverage_df.to_csv(os.path.join(self.output_dir, "metric_coverage.csv"), index=False)
        numeric_cols = [c for c in numeric_cols if coverage_df[coverage_df["metric"] == c]["coverage_ok"].values[0]]
        stats = df[numeric_cols].describe().transpose()
        # print("Descriptive statistics saved.")
        
        # 2. Correlation Matrix
        corr = df[numeric_cols].corr()
        # plot_correlation_matrix removed per visualization policy
        
        try:
            means = df[numeric_cols].mean()
            inf = [
                means.get("normalized_levenshtein", 0),
                means.get("char_ngram_jaccard", 0),
                abs(means.get("compression_delta", 0)),
            ]
            tok = [
                abs(means.get("token_count_change", 0)),
                means.get("fragmentation_index", 0),
                abs(means.get("normalized_entropy_delta", 0)),
            ]
            sem = [
                means.get("contextual_embedding_distance", 0),
                1.0 - means.get("semantic_entailment_score", 0),
                means.get("lm_surprisal_delta", 0),
            ]
            syn = [
                1.0 - means.get("dependency_overlap_score", 0),
                abs(means.get("tree_depth_change", 0)),
                means.get("pos_divergence", 0),
            ]
            rhy = [
                abs(means.get("syllable_count_change", 0)),
                means.get("stress_pattern_divergence", 0),
            ]
            vis = [
                means.get("ssim_distance", 0),
                abs(means.get("glyph_displacement", 0)),
                means.get("spatial_dispersion_salience_score", 0),
            ]
            profile = {
                "Informatics": float(np.nanmean(inf)),
                "Tokenization": float(np.nanmean(tok)),
                "Semantics": float(np.nanmean(sem)),
                "Syntax": float(np.nanmean(syn)),
                "Rhythm": float(np.nanmean(rhy)),
                "Visualization": float(np.nanmean(vis)),
            }
            vals = [profile[k] for k in ["Informatics", "Tokenization", "Semantics", "Syntax", "Rhythm", "Visualization"]]
            cats = ["Informatics", "Tokenization", "Semantics", "Syntax", "Rhythm", "Visualization"]
            plot_holistic_radar(cats, vals, "Holistic Perturbation Profile", os.path.join(self.output_dir, "holistic_perturbation_profile.png"))
        except Exception:
            profile = {"General": 0.0}
        
        # 4. Group Comparison (Typo vs Whitespace): Welch's t-test, Cohen's d, 95% CI
        comparisons = []
        for col in numeric_cols:
            a = df[df['type'] == 'Typo'][col].dropna()
            b = df[df['type'] == 'Whitespace'][col].dropna()
            if len(a) >= 3 and len(b) >= 3:
                m1, m2 = a.mean(), b.mean()
                s1, s2 = a.std(ddof=1), b.std(ddof=1)
                n1, n2 = len(a), len(b)
                diff = m1 - m2
                se = np.sqrt((s1**2 / n1) + (s2**2 / n2))
                low_var = (np.ptp(a) < eps and np.ptp(b) < eps) or se < eps or (s1 < eps and s2 < eps)
                if not low_var:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", "Precision loss occurred in moment calculation due to catastrophic cancellation.*", RuntimeWarning)
                        tstat, pval = sstats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
                else:
                    try:
                        mw = sstats.mannwhitneyu(a, b, alternative='two-sided')
                        tstat = np.nan
                        pval = mw.pvalue
                    except Exception:
                        tstat = np.nan
                        pval = np.nan
                # Cohen's d (pooled SD using unbiased estimator)
                sp2 = (((n1 - 1) * s1**2) + ((n2 - 1) * s2**2)) / (n1 + n2 - 2) if (n1 + n2 - 2) > 0 else np.nan
                pooled = np.sqrt(sp2) if sp2 >= 0 else np.nan
                d = diff / pooled if pooled and pooled > eps else np.nan
                try:
                    J = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
                    hedges_g = d * J if np.isfinite(d) else np.nan
                except Exception:
                    hedges_g = np.nan
                try:
                    A = a.values.reshape(-1, 1)
                    B = b.values.reshape(1, -1)
                    comp = A - B
                    wins = np.sum(comp > 0)
                    losses = np.sum(comp < 0)
                    cliffs_delta = (wins - losses) / float(n1 * n2)
                except Exception:
                    cliffs_delta = np.nan
                # 95% CI for diff using Welch-Satterthwaite
                df_welch_num = (s1**2 / n1 + s2**2 / n2)**2
                df_welch_den = ((s1**2 / n1)**2 / (n1 - 1)) + ((s2**2 / n2)**2 / (n2 - 1))
                df_welch = df_welch_num / df_welch_den if df_welch_den > 0 else np.nan
                if not low_var and se > eps:
                    try:
                        tcrit = sstats.t.ppf(0.975, df=df_welch)
                        ci_low = diff - tcrit * se
                        ci_high = diff + tcrit * se
                    except Exception:
                        ci_low = np.nan
                        ci_high = np.nan
                else:
                    ci_low = np.nan
                    ci_high = np.nan
                comparisons.append({
                    'metric': col,
                    'mean_typo': m1,
                    'mean_whitespace': m2,
                    'diff': diff,
                    'cohen_d': d,
                    'hedges_g': hedges_g,
                    'cliffs_delta': cliffs_delta,
                    'p_value': pval,
                    'ci_low': ci_low,
                    'ci_high': ci_high
                })
        _comp_cols = ['metric', 'mean_typo', 'mean_whitespace', 'diff', 'cohen_d', 'hedges_g', 'cliffs_delta', 'p_value', 'ci_low', 'ci_high']
        group_comp = pd.DataFrame(comparisons, columns=_comp_cols).sort_values(by='p_value', ascending=True)
        try:
            if not group_comp.empty:
                order = group_comp['metric'].tolist()
                stats = stats.reindex(order)
        except Exception:
            pass
        stats.to_csv(os.path.join(self.output_dir, "descriptive_stats.csv"))
        group_comp.to_csv(os.path.join(self.output_dir, "group_comparison.csv"), index=False)
        
        # Generate reports
        self.reporter.generate(df, stats, profile, group_comp=group_comp, sample_size=len(df))
        print(f"Report draft generated at: {os.path.join(self.output_dir, 'SCIENTIFIC_REPORT_DRAFT.md')}")
        try:
            self.reporter.generate_pptx(df, stats, profile, group_comp=group_comp)
            print(f"PPTX report generated at: {os.path.join(self.output_dir, 'SCIENTIFIC_REPORT_DRAFT.pptx')}")
        except Exception as e:
            print(f"Failed to generate PPTX report: {e}")
        try:
            self.reporter.generate_docx(df, stats, profile, group_comp=group_comp)
            print(f"WORD report generated at: {os.path.join(self.output_dir, 'SCIENTIFIC_REPORT_DRAFT.docx')}")
        except Exception as e:
            print(f"Failed to generate WORD report: {e}")
        
        # print("Analysis complete. Results saved to", self.output_dir)
        
        try:
            from project.utils.plotting import plot_influence_heatmaps
            plot_influence_heatmaps(group_comp, os.path.join(self.output_dir, "influence"))
        except Exception:
            pass
        # Multiple Hypothesis Testing Correction:
        # Since we test ~20 independent metrics, the family-wise error rate increases.
        # We apply the Benjamini-Hochberg (BH) procedure to control the False Discovery Rate (FDR).
        # q-values (adjusted p-values) < 0.05 indicate significant discovery after correction.
        try:
            m = len(group_comp)
            pvals = group_comp['p_value'].values
            order_idx = np.argsort(pvals)
            qvals = np.empty(m)
            ranks = np.empty(m, dtype=int)
            ranks[order_idx] = np.arange(1, m + 1)
            q = pvals * m / ranks
            q = np.minimum.accumulate(q[order_idx][::-1])[::-1]
            inv = np.empty_like(order_idx)
            inv[order_idx] = np.arange(m)
            qvals = q[inv]
            group_comp['q_value'] = qvals
        except Exception:
            group_comp['q_value'] = np.nan
        group_comp.to_csv(os.path.join(self.output_dir, "group_comparison.csv"), index=False)

        
        # Skip whitespace subgroup analysis outputs per visualization policy
        
        # 6. Within-type significance vs 0
        def one_sample_table(subdf: pd.DataFrame, label: str) -> pd.DataFrame:
            rows = []
            for col in numeric_cols:
                x = subdf[col].dropna()
                if len(x) >= 3:
                    m = x.mean()
                    sd = x.std(ddof=1)
                    n = len(x)
                    if sd > eps and np.ptp(x) >= eps:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", "Precision loss occurred in moment calculation due to catastrophic cancellation.*", RuntimeWarning)
                            tstat, pval = sstats.ttest_1samp(x, 0.0, nan_policy='omit')
                        d = m / sd if sd and sd > eps else np.nan
                    else:
                        try:
                            if np.any(np.abs(x) > eps):
                                w = sstats.wilcoxon(x, zero_method='wilcox', alternative='two-sided', correction=False, mode='auto')
                                tstat = np.nan
                                pval = w.pvalue
                            else:
                                tstat = 0.0
                                pval = 1.0
                            d = np.nan
                        except Exception:
                            tstat = np.nan
                            pval = np.nan
                            d = np.nan
                    rows.append({'metric': col, 'mean': m, 'std': sd, 'n': n, 't': tstat, 'p_value': pval, 'cohen_d': d})
            return pd.DataFrame(rows, columns=['metric','mean','std','n','t','p_value','cohen_d']).sort_values(by='p_value', ascending=True)
        typo_signif = one_sample_table(df[df['type'] == 'Typo'], 'Typo')
        ws_signif = one_sample_table(df[df['type'] == 'Whitespace'], 'Whitespace')
        typo_signif.to_csv(os.path.join(self.output_dir, "significance_typo.csv"), index=False)
        ws_signif.to_csv(os.path.join(self.output_dir, "significance_whitespace.csv"), index=False)
        
        # Skip effect bars per visualization policy
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import roc_auc_score, accuracy_score
            import json as _json
            y = (df['type'] == 'Typo').astype(int)
            X = df[numeric_cols].copy()
            X = X.apply(pd.to_numeric, errors='coerce')
            X = X.fillna(X.mean())
            X = (X - X.mean()) / (X.std(ddof=0) + eps)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
            model_name = ""
            clf = None
            spec_xgb = importlib.util.find_spec("xgboost")
            if spec_xgb is not None:
                xgb = importlib.import_module("xgboost")
                params = dict(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42)
                gpu_used = False
                try:
                    ver = getattr(xgb, "__version__", "0")
                    major = int(ver.split(".")[0]) if ver else 0
                    if major >= 2:
                        try:
                            clf = xgb.XGBClassifier(**params, tree_method="hist", device="cuda")
                            gpu_used = True
                        except Exception:
                            clf = xgb.XGBClassifier(**params)
                    else:
                        try:
                            clf = xgb.XGBClassifier(**params, tree_method="gpu_hist", predictor="gpu_predictor")
                            gpu_used = True
                        except Exception:
                            clf = xgb.XGBClassifier(**params)
                except Exception:
                    clf = xgb.XGBClassifier(**params)
                model_name = "XGBoost(GPU)" if gpu_used else "XGBoost"
            else:
                from sklearn.ensemble import GradientBoostingClassifier
                clf = GradientBoostingClassifier(random_state=42)
                model_name = "GradientBoosting"
            clf.fit(X_train, y_train)
            try:
                if model_name.startswith("XGBoost"):
                    booster = clf.get_booster()
                    used_gpu = model_name.startswith("XGBoost(GPU)")
                    if used_gpu:
                        try:
                            import cupy as _cp  # type: ignore
                            booster.set_param({'device': 'cuda'})
                            arr = _cp.asarray(X_test.values)
                            y_proba = booster.inplace_predict(arr)
                        except Exception:
                            import warnings as _warn
                            _warn.filterwarnings("ignore", category=UserWarning, message=".*Falling back to prediction using DMatrix due to mismatched devices.*")
                            dtest = xgb.DMatrix(X_test.values)
                            y_proba = booster.predict(dtest)
                    else:
                        dtest = xgb.DMatrix(X_test.values)
                        y_proba = booster.predict(dtest)
                else:
                    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
            except Exception:
                if model_name.startswith("XGBoost"):
                    booster = clf.get_booster()
                    dtest = xgb.DMatrix(X_test.values)
                    y_proba = booster.predict(dtest)
                else:
                    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
            auc_ml = roc_auc_score(y_test, y_proba)
            acc_ml = accuracy_score(y_test, (y_proba >= 0.5).astype(int))
            ml_info = {"model": model_name, "roc_auc": float(auc_ml), "accuracy": float(acc_ml)}
            with open(os.path.join(self.output_dir, "ml_results.json"), "w", encoding="utf-8") as f:
                _json.dump(ml_info, f)
            shap_ok = False
            spec_shap = importlib.util.find_spec("shap")
            if spec_shap is not None:
                shap = importlib.import_module("shap")
                shap_ok = True
                try:
                    clf_for_shap = clf
                    if model_name.startswith("XGBoost(GPU)"):
                        try:
                            model_file = os.path.join(self.output_dir, "xgb_model.json")
                            clf.get_booster().save_model(model_file)
                            clf_cpu = xgb.XGBClassifier(**params)
                            try:
                                clf_cpu.set_params(device="cpu", tree_method="hist")
                            except Exception:
                                pass
                            clf_cpu.load_model(model_file)
                            clf_for_shap = clf_cpu
                        except Exception:
                            clf_for_shap = clf
                    try:
                        explainer = shap.TreeExplainer(clf_for_shap)
                        shap_values = explainer.shap_values(X_test.values)
                        vals = shap_values if isinstance(shap_values, (list, tuple)) else shap_values
                    except Exception:
                        explainer = shap.Explainer(clf_for_shap, X_train.values)
                        explanation = explainer(X_test.values)
                        vals = explanation.values
                    import numpy as _np
                    imp = _np.mean(_np.abs(vals), axis=0)
                    shap_df = pd.DataFrame({"metric": X.columns.tolist(), "mean_abs_shap": imp})
                    shap_df = shap_df.sort_values(by="mean_abs_shap", ascending=False)
                    shap_df.to_csv(os.path.join(self.output_dir, "shap_importance.csv"), index=False)
                    try:
                        configure_nature_style()
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=FutureWarning, message=".*NumPy global RNG.*")
                            warnings.filterwarnings("ignore", category=UserWarning, message=".*figure layout has changed to tight.*")
                            try:
                                explanation = None
                                try:
                                    explanation = shap.Explanation(values=vals, base_values=None, data=X_test.values, feature_names=X.columns.tolist())
                                except Exception:
                                    explanation = None
                                if explanation is not None and hasattr(shap, 'plots') and hasattr(shap.plots, 'bar'):
                                    shap.plots.bar(explanation, max_display=SHAP_MAX_DISPLAY_BAR, show=False)
                                else:
                                    shap.summary_plot(vals, X_test.values, feature_names=X.columns.tolist(), plot_type="bar", show=False, max_display=SHAP_MAX_DISPLAY_BAR)
                            except Exception:
                                shap.summary_plot(vals, X_test.values, feature_names=X.columns.tolist(), plot_type="bar", show=False, max_display=SHAP_MAX_DISPLAY_BAR)
                        plt.tight_layout()
                        p = os.path.join(self.output_dir, "shap_summary_bar.png")
                        plt.savefig(p, bbox_inches="tight", dpi=600)
                        try:
                            plt.savefig(os.path.splitext(p)[0] + ".svg", bbox_inches="tight", format="svg")
                        except Exception:
                            pass
                        print("Saved SHAP bar summary to shap_summary_bar.png")
                        plt.close()
                    except Exception:
                        pass
                    try:
                        configure_nature_style()
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=FutureWarning, message=".*NumPy global RNG.*")
                            warnings.filterwarnings("ignore", category=UserWarning, message=".*figure layout has changed to tight.*")
                            try:
                                explanation = None
                                try:
                                    explanation = shap.Explanation(values=vals, base_values=None, data=X_test.values, feature_names=X.columns.tolist())
                                except Exception:
                                    explanation = None
                                if explanation is not None and hasattr(shap, 'plots') and hasattr(shap.plots, 'beeswarm'):
                                    shap.plots.beeswarm(explanation, max_display=SHAP_MAX_DISPLAY_SWARM, show=False)
                                else:
                                    shap.summary_plot(vals, X_test.values, feature_names=X.columns.tolist(), show=False, max_display=SHAP_MAX_DISPLAY_SWARM)
                            except Exception:
                                shap.summary_plot(vals, X_test.values, feature_names=X.columns.tolist(), show=False, max_display=SHAP_MAX_DISPLAY_SWARM)
                        plt.tight_layout()
                        p = os.path.join(self.output_dir, "shap_summary.png")
                        plt.savefig(p, bbox_inches="tight", dpi=600)
                        try:
                            plt.savefig(os.path.splitext(p)[0] + ".svg", bbox_inches="tight", format="svg")
                        except Exception:
                            pass
                        print("Saved SHAP summary (beeswarm) to shap_summary.png")
                        plt.close()
                    except Exception:
                        pass
                    try:
                        std_ok = [f for f in X.columns.tolist() if (X_test[f].std(ddof=0) > eps and np.ptp(X_test[f].values) > eps)]
                        top_feat = [f for f in shap_df['metric'].tolist() if f in std_ok][:2]
                        for feat in top_feat:
                            configure_nature_style()
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", category=FutureWarning, message=".*NumPy global RNG.*")
                                warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in divide.*")
                                shap.dependence_plot(feat, vals, X_test.values, feature_names=X.columns.tolist(), interaction_index='auto', show=False)
                            plt.tight_layout()
                            p = os.path.join(self.output_dir, f"shap_dependence_{feat}.png")
                            plt.savefig(p, bbox_inches="tight", dpi=600)
                            try:
                                plt.savefig(os.path.splitext(p)[0] + ".svg", bbox_inches="tight", format="svg")
                            except Exception:
                                pass
                            print(f"Saved SHAP dependence plot for {feat}")
                            plt.close()
                    except Exception:
                        pass
                except Exception:
                    shap_ok = False
            else:
                shap_ok = False
            if not shap_ok:
                try:
                    from sklearn.inspection import permutation_importance
                    r = permutation_importance(clf, X_test, y_test, n_jobs=1, random_state=42)
                    imp_df = pd.DataFrame({"metric": X.columns.tolist(), "importance": r.importances_mean})
                    imp_df = imp_df.sort_values(by="importance", ascending=False)
                    imp_df.to_csv(os.path.join(self.output_dir, "permutation_importance.csv"), index=False)
                except Exception:
                    pass
        except Exception:
            pass
        
        # Nature MI Figure 4-style compound visualization removed per user request
        # The logic for plot_compound_overview and figure4_compound.png generation is deleted.
        
        # Skip single-metric distribution plots per visualization policy
        dist_images = None
        top_metrics = group_comp['metric'].head(3).tolist()
        # Unified output policy: keep all artifacts directly under `self.output_dir`
        # No moving/copying to subdirectories to avoid duplicates
        
        try:
            category_order = ["Informatics", "Tokenization", "Semantics", "Syntax", "Rhythm", "Visualization"]
            category_map = {
                "normalized_levenshtein": "Informatics",
                "char_ngram_jaccard": "Informatics",
                "compression_delta": "Informatics",
                "token_count_change": "Tokenization",
                "fragmentation_index": "Tokenization",
                "normalized_entropy_delta": "Tokenization",
                "contextual_embedding_distance": "Semantics",
                "semantic_entailment_score": "Semantics",
                "lm_surprisal_delta": "Semantics",
                "dependency_overlap_score": "Syntax",
                "tree_depth_change": "Syntax",
                "pos_divergence": "Syntax",
                "syllable_count_change": "Rhythm",
                "stress_pattern_divergence": "Rhythm",
                "ssim_distance": "Visualization",
                "glyph_displacement": "Visualization",
                "spatial_dispersion_salience_score": "Visualization",
            }
            C = corr.copy()
            labels = [category_map.get(c, "Unknown") for c in C.columns]
            n = len(labels)
            vals = C.values
            intra = []
            inter = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not np.isnan(vals[i, j]):
                        if labels[i] == labels[j]:
                            intra.append(abs(vals[i, j]))
                        else:
                            inter.append(abs(vals[i, j]))
            intra_mean = float(np.nanmean(intra)) if len(intra) > 0 else float('nan')
            inter_mean = float(np.nanmean(inter)) if len(inter) > 0 else float('nan')
            delta = intra_mean - inter_mean if np.isfinite(intra_mean) and np.isfinite(inter_mean) else float('nan')
            rng = np.random.default_rng(42)
            perm = []
            if len(intra) > 0 and len(inter) > 0:
                for _ in range(1000):
                    lbl = rng.permutation(labels)
                    p_intra = []
                    p_inter = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if not np.isnan(vals[i, j]):
                                if lbl[i] == lbl[j]:
                                    p_intra.append(abs(vals[i, j]))
                                else:
                                    p_inter.append(abs(vals[i, j]))
                    m_in = float(np.nanmean(p_intra)) if len(p_intra) > 0 else float('nan')
                    m_out = float(np.nanmean(p_inter)) if len(p_inter) > 0 else float('nan')
                    if np.isfinite(m_in) and np.isfinite(m_out):
                        perm.append(m_in - m_out)
            if len(perm) > 10 and np.isfinite(delta):
                p_value = float((np.sum(np.abs(perm) >= abs(delta)) + 1) / (len(perm) + 1))
            else:
                p_value = float('nan')
            grp_stats = []
            for g in category_order:
                idx = [k for k, c in enumerate(C.columns) if category_map.get(c) == g]
                v = []
                for i in range(len(idx)):
                    for j in range(i + 1, len(idx)):
                        a = idx[i]
                        b = idx[j]
                        if not np.isnan(vals[a, b]):
                            v.append(abs(vals[a, b]))
                m = float(np.nanmean(v)) if len(v) > 0 else float('nan')
                grp_stats.append({"group": g, "intra_mean_abs_corr": m, "count_pairs": len(v)})
            out_json = {
                "intra_mean_abs_corr": intra_mean,
                "inter_mean_abs_corr": inter_mean,
                "delta_intra_minus_inter": delta,
                "permutation_p_value": p_value,
                "group_stats": grp_stats
            }
            with open(os.path.join(self.output_dir, "cluster_consistency.json"), "w", encoding="utf-8") as f:
                json.dump(out_json, f, ensure_ascii=False, indent=2)
            with open(os.path.join(self.output_dir, "cluster_consistency.md"), "w", encoding="utf-8") as f:
                f.write("## Cluster Consistency Audit\n\n")
                f.write(f"- Intra-group mean|corr|: {intra_mean:.3f}\n")
                f.write(f"- Inter-group mean|corr|: {inter_mean:.3f}\n")
                f.write(f"- Separation (Δ): {delta:.3f}\n")
                f.write(f"- Permutation p-value: {p_value:.3f}\n\n")
                for s in grp_stats:
                    f.write(f"- {s['group']}: intra mean|corr|={s['intra_mean_abs_corr']:.3f} (pairs={s['count_pairs']})\n")
        except Exception:
            pass
        
        # Modern Multivariate: Clustermap & Anomaly Detection
        try:
            from project.utils.plotting import plot_metric_clustermap, plot_metric_ridgeline
            category_order = ["Informatics", "Tokenization", "Semantics", "Syntax", "Rhythm", "Visualization"]
            category_map = {
                "normalized_levenshtein": "Informatics",
                "char_ngram_jaccard": "Informatics",
                "compression_delta": "Informatics",
                "token_count_change": "Tokenization",
                "fragmentation_index": "Tokenization",
                "normalized_entropy_delta": "Tokenization",
                "contextual_embedding_distance": "Semantics",
                "semantic_entailment_score": "Semantics",
                "lm_surprisal_delta": "Semantics",
                "dependency_overlap_score": "Syntax",
                "tree_depth_change": "Syntax",
                "pos_divergence": "Syntax",
                "syllable_count_change": "Rhythm",
                "stress_pattern_divergence": "Rhythm",
                "ssim_distance": "Visualization",
                "glyph_displacement": "Visualization",
                "spatial_dispersion_salience_score": "Visualization",
            }
            plot_metric_clustermap(corr, os.path.join(self.output_dir, "metric_taxonomy_clustermap.png"), category_map=category_map, category_order=category_order, cbar_loc="left", label_style="text")
            plot_metric_ridgeline(df, os.path.join(self.output_dir, "metric_ridgeline_plot.png"), category_map=category_map, category_order=category_order)
            
            # Anomaly Detection (Mahalanobis proxy)
            cols_var = [c for c in numeric_cols if pd.to_numeric(df[c], errors='coerce').std(ddof=0) > eps]
            X_anom = df[cols_var].dropna()
            if len(X_anom) > 20:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_anom)
                import numpy as _np
                # Euclidean distance from centroid (0,0...0) in scaled space ~ Mahalanobis
                dists = _np.linalg.norm(X_scaled, axis=1)
                df.loc[X_anom.index, 'anomaly_score'] = dists
                top_anomalies = df.loc[X_anom.index].sort_values('anomaly_score', ascending=False).head(10)
                top_anomalies[['type', 'original', 'disturbed', 'anomaly_score']].to_csv(os.path.join(self.output_dir, "top_anomalies.csv"))
        except Exception:
            pass

        # Generate Scientific Report Draft
        self.reporter.generate(df, stats, profile, group_comp=group_comp, sample_size=len(df))
        # Generate PPTX Draft
        self.reporter.generate_pptx(df, stats, profile, group_comp, distribution_images=dist_images)
        # Generate DOCX Report
        try:
            self.reporter.generate_docx(df, stats, profile, group_comp, sample_size=len(df))
        except Exception:
            pass
        # Skip compound overview per visualization policy

        print(f"Analysis complete. Results saved to {self.output_dir}")
        

        # Skip AUC discriminative bar and traditional PCA per visualization policy
        try:
            pass
        except Exception:
            pass
        try:
            figure_quality_report(self.output_dir, outfile=os.path.join(self.output_dir, "figure_quality.csv"))
            generate_qc_report(
                self.output_dir,
                outfile_md=os.path.join(self.output_dir, "figure_qc_report.md"),
                outfile_csv=os.path.join(self.output_dir, "figure_qc.csv"),
                min_font=MIN_FONT
            )
        except Exception:
            pass

        # Write experiment meta
        try:
            import json as _json
            meta = {}
            meta["dataset_size"] = int(len(df))
            import datetime as _dt
            meta["finished_at"] = _dt.datetime.now().isoformat(timespec="seconds")
            # GPU info
            try:
                import torch as _torch
            except Exception:
                _torch = None
            meta["gpu_available"] = bool(_torch and _torch.cuda.is_available())
            if meta["gpu_available"]:
                try:
                    c = _torch.cuda.device_count()
                    names = [ _torch.cuda.get_device_name(i) for i in range(c) ]
                except Exception:
                    names = []
                meta["gpu_devices"] = names
            else:
                meta["gpu_devices"] = []
            # Model info
            try:
                ml_json = os.path.join(self.output_dir, "ml_results.json")
                if os.path.exists(ml_json):
                    with open(ml_json, "r", encoding="utf-8") as f:
                        info = _json.load(f)
                    meta["model"] = info.get("model")
                else:
                    meta["model"] = None
            except Exception:
                meta["model"] = None
            # Seeds
            meta["seeds"] = {"random": 42, "numpy": 42}
            # Versions
            meta["versions"] = {}
            try:
                import numpy as _np
                import pandas as _pd
                import sklearn as _sk
                meta["versions"]["numpy"] = getattr(_np, "__version__", None)
                meta["versions"]["pandas"] = getattr(_pd, "__version__", None)
                meta["versions"]["sklearn"] = getattr(_sk, "__version__", None)
            except Exception:
                pass
            try:
                import xgboost as _xgb
                meta["versions"]["xgboost"] = getattr(_xgb, "__version__", None)
            except Exception:
                meta["versions"]["xgboost"] = None
            try:
                import shap as _shap
                meta["versions"]["shap"] = getattr(_shap, "__version__", None)
            except Exception:
                meta["versions"]["shap"] = None
            with open(os.path.join(self.output_dir, "experiment_meta.json"), "w", encoding="utf-8") as f:
                _json.dump(meta, f)
        except Exception:
            pass

        # Outputs already organized above; skip duplication

def _base_sentences():
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence enables new forms of scientific discovery.",
        "Data quality determines the reliability of statistical conclusions.",
        "Language models learn patterns from large corpora of text.",
        "Whitespace characters can alter tokenization boundaries silently.",
        "Misspellings may degrade readability but preserve core semantics.",
        "Robust metrics should be consistent across perturbation types.",
        "Careful visualization improves interpretability and trust.",
        "Entropy captures distributional changes in character sequences.",
        "Compression reflects structural redundancy and variability.",
        "Tokenization affects downstream parsing and tagging behavior.",
        "Contextual embeddings encode semantic information about words.",
        "Surprisal measures model uncertainty for given sequences.",
        "Syllable changes can influence prosodic characteristics.",
        "Dependency structures capture syntactic relationships.",
        "Glyph layout differences reflect visual rendering changes.",
        "Effect sizes quantify practical significance beyond p-values.",
        "Confidence intervals indicate estimation uncertainty ranges.",
        "Cohen's d normalizes mean differences by pooled variance.",
        "Scientific reports require reproducible experimental design.",
        "Statistical power increases with larger sample sizes.",
        "Outliers can bias estimators and mislead interpretations.",
        "Normalization facilitates fair comparison across metrics.",
        "Triangular heatmaps remove redundant correlation information.",
        "Accessible color palettes improve figure readability.",
        "Panel lettering supports multi-part figure organization.",
        "Model explanations guide mechanism-level understanding.",
        "Feature importance reveals discriminative signals.",
        "Prosody relates to stress and rhythm in language.",
        "Parsing identifies syntactic roles and dependencies.",
        "Token boundaries shape lexical segmentation and analysis.",
        "Character edits change local n-gram structure.",
        "Semantic drift occurs when meaning shifts subtly.",
        "Zero-width spaces are invisible yet impactful.",
        "Non-breaking spaces prevent unwanted line wraps.",
        "Adjacent transpositions are common typographical errors.",
        "Vowel substitutions can form plausible variants.",
        "Deterministic generation reduces experimental variance.",
        "Reproducible pipelines enhance scientific credibility.",
        "Vector graphics preserve detail at any scale.",
        "Triangular layout emphasizes unique correlations.",
        "Granular metrics expose targeted disturbance effects.",
        "Holistic profiles summarize multi-dimensional behavior.",
        "Quality checks enforce publication-grade standards.",
        "Batch studies aggregate evidence across samples.",
        "Correlation matrices reveal metric relationships.",
        "Diverging colormaps highlight directionality of effects.",
        "Tight layout improves information density in figures.",
        "Consistent typography supports visual harmony.",
        "Structured analysis leads to actionable insights."
    ]

def _template_bank():
    cats = [
        ("Science", ["researchers", "scientists", "analysts"], ["investigate", "evaluate", "quantify"], ["hypotheses", "models", "datasets"]),
        ("Technology", ["engineers", "developers", "systems"], ["deploy", "optimize", "monitor"], ["algorithms", "applications", "services"]),
        ("Health", ["clinicians", "patients", "hospitals"], ["report", "assess", "improve"], ["outcomes", "therapies", "records"]),
        ("Finance", ["markets", "investors", "auditors"], ["react", "forecast", "audit"], ["risks", "returns", "statements"]),
        ("Education", ["teachers", "students", "schools"], ["adopt", "learn", "evaluate"], ["curricula", "skills", "exams"]),
        ("Environment", ["ecologists", "rangers", "agencies"], ["track", "preserve", "regulate"], ["habitats", "emissions", "resources"]),
        ("Industry", ["manufacturers", "companies", "factories"], ["produce", "inspect", "ship"], ["components", "products", "orders"]),
        ("Media", ["journalists", "editors", "publishers"], ["curate", "verify", "publish"], ["articles", "sources", "stories"]),
        ("Transport", ["drivers", "airlines", "networks"], ["plan", "schedule", "coordinate"], ["routes", "flights", "deliveries"]),
        ("PublicPolicy", ["governments", "agencies", "committees"], ["draft", "review", "enforce"], ["regulations", "policies", "standards"]),
    ]
    templates = []
    for cat, subs, verbs, objs in cats:
        for i in range(len(subs)):
            s = f"{subs[i].capitalize()} {verbs[i]} {objs[i]} to achieve reliable results."
            templates.append({"text": s, "category": cat})
    # add base corpus items with category labels
    for s in _base_sentences():
        templates.append({"text": s, "category": "General"})
    return templates

def _variant_sentence(base_text, variant_idx):
    if variant_idx % 3 == 0:
        return base_text
    elif variant_idx % 3 == 1:
        return base_text + " In practice, rigorous validation supports trustworthy conclusions."
    else:
        return base_text + " Furthermore, cross-domain evidence and pre-registered protocols strengthen reliability across contexts."

def _choose_token_idx(tokens):
    n = len(tokens)
    if n == 0:
        return 0
    s = "".join(tokens)
    h = sum(ord(c) for c in s) % n
    return h

def _apply_typo(s):
    tokens = s.split()
    if not tokens:
        return s
    idx = _choose_token_idx(tokens)
    t = tokens[idx]
    
    # Single perturbation logic
    if t.isalpha() and len(t) >= 4:
        pos = max(1, len(t) // 2)
        chars = list(t)
        chars[pos - 1], chars[pos] = chars[pos], chars[pos - 1]
        t = "".join(chars)
    else:
        m = re.search(r"[aeiouAEIOU]", t)
        if m:
            v = m.group(0)
            sub = {"a": "e", "e": "a", "i": "e", "o": "a", "u": "o", "A": "E", "E": "A", "I": "E", "O": "A", "U": "O"}
            t = t.replace(v, sub.get(v, v), 1)
        else:
            t = t + "x"
            
    tokens[idx] = t
    return " ".join(tokens)

def _apply_whitespace(s):
    zw = "\u200b"
    tokens = s.split()
    if not tokens:
        return s, zw, "U+200B"
    idx = _choose_token_idx(tokens)
    t = tokens[idx]
    
    # Single perturbation logic
    if len(t) >= 2:
        inject = t
        inject = inject[:2] + zw + inject[2:]
        tokens[idx] = inject
        return " ".join(tokens), zw, "U+200B"
        
    if len(tokens) >= 2 and idx < len(tokens) - 1:
        inject = tokens[idx]
        inject = inject + zw + tokens[idx + 1]
        tokens[idx] = inject
        del tokens[idx + 1]
        return " ".join(tokens), zw, "U+200B"
        
    return s + zw, zw, "U+200B"

def _length_bin(s):
    L = len(s)
    if L < 70:
        return "short"
    elif L < 120:
        return "medium"
    else:
        return "long"

def build_high_quality_dataset(n=500):
    """
    Build a dataset from the AdvBench parallel corpus (real perturbation pairs).

    The corpus is treated as the primary empirical benchmark: no synthetic perturbations are
    created here. The resulting dataset is deterministically ordered to support reproducibility.

    Args:
        n: Maximum number of pairs to load. If None, loads all available items.

    Returns:
        A list of dataset items compatible with BatchStudy.run_batch().
    """
    # Use relative path to project root to ensure portability across environments
    # project/study.py -> dirname -> project -> dirname -> Metrics (Root)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, "Corpus", "advbench_unicode_whitespace_perturbed.json")
    pairs = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            limit = len(data) if n is None else min(n, len(data))
            for i in range(limit):
                item = data[i]
                s = str(item.get("goal", ""))
                sp = str(item.get("perturbed_text", ""))
                if not s or not sp:
                    continue
                pairs.append({
                    "original": s,
                    "disturbed": sp,
                    "type": "Whitespace",
                    "category": "AdvBench",
                    "length_bin": _length_bin(s),
                    "ws_char": None,
                    "ws_code": None
                })
        else:
            pairs = []
    except Exception as e:
        print(f"Error loading dataset: {e}")
        pairs = []
    
    # Deterministic ordering for reproducibility
    if pairs:
        pairs = sorted(pairs, key=lambda r: (r["type"], r.get("category",""), r.get("length_bin",""), sum(ord(c) for c in r["original"])))
    
    return pairs



def summarize_dataset(pairs, outdir):
    """
    Persist basic dataset composition summaries for audit and reporting.

    Writes both an Excel workbook and CSV tables describing the distribution of samples by:
    - perturbation type
    - semantic/category label
    - length bin

    Args:
        pairs: Dataset items as produced by build_high_quality_dataset().
        outdir: Output directory where the summary files are written.
    """
    try:
        df = pd.DataFrame(pairs)
        cnt_type = df.groupby("type")["original"].count().rename("count").reset_index()
        cnt_cat = df.groupby(["type","category"])["original"].count().rename("count").reset_index()
        cnt_len = df.groupby(["type","length_bin"])["original"].count().rename("count").reset_index()
        with pd.ExcelWriter(os.path.join(outdir, "dataset_summary.xlsx")) as w:
            cnt_type.to_excel(w, sheet_name="type_counts", index=False)
            cnt_cat.to_excel(w, sheet_name="category_counts", index=False)
            cnt_len.to_excel(w, sheet_name="length_counts", index=False)
        cnt_type.to_csv(os.path.join(outdir, "dataset_type_counts.csv"), index=False)
        cnt_cat.to_csv(os.path.join(outdir, "dataset_category_counts.csv"), index=False)
        cnt_len.to_csv(os.path.join(outdir, "dataset_length_counts.csv"), index=False)
    except Exception:
        pass
# Example Usage
if __name__ == "__main__":
    # Register all metrics before running study
    metric_skills.register_all_metrics()
    
    study = BatchStudy()
    dataset_500 = build_high_quality_dataset(500)
    summarize_dataset(dataset_500, study.output_dir)
    df = study.run_batch(dataset_500)
    study.analyze_results(df)
