"""
Wavelet-Based Perturbation Dynamics Analysis.

This module provides a signal-processing view of textual perturbations. It converts texts into
aligned 1D signals (preferably via shared PCA projection of token embeddings) and computes a
continuous wavelet transform (CWT) to obtain a time–scale energy representation. Differences
between original and perturbed scalograms serve as a compact "dynamics signature".

Scientific Rationale:
- Wavelets capture localized, multi-scale changes, which is useful for perturbations that affect
  token boundaries or introduce sparse, high-frequency artifacts (e.g., Unicode whitespace).
- A shared projection space (PCA fitted on combined embeddings) improves comparability between
  the original and perturbed sequences.
"""

import numpy as np
import pandas as pd
from scipy import signal
import warnings

_HF_MODEL_CACHE = {}
_HF_TOKENIZER_CACHE = {}

def _get_hf_tokenizer(model_name: str):
    """
    Retrieve a Hugging Face tokenizer with in-memory caching.

    Behaviour:
    - Uses _HF_TOKENIZER_CACHE to avoid repeated construction of the same tokenizer.
    - If the tokenizer has no pad_token but does have an eos_token, we set pad_token
      to eos_token to prevent downstream tensor shape issues.

    Args:
        model_name: model identifier, e.g. "bert-base-uncased".

    Returns:
        A tokenizer instance compatible with the given model.
    """
    tok = _HF_TOKENIZER_CACHE.get(model_name)
    if tok is not None:
        return tok
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        try:
            tok.pad_token = tok.eos_token
        except Exception:
            pass
    # Ensure a reasonable model_max_length to avoid truncation warnings
    try:
        ml = getattr(tok, "model_max_length", None)
        if not isinstance(ml, int) or ml is None or ml > 4096:
            tok.model_max_length = 512
    except Exception:
        try:
            tok.model_max_length = 512
        except Exception:
            pass
    _HF_TOKENIZER_CACHE[model_name] = tok
    return tok

def _get_hf_model(model_name: str, device: str | None = None):
    """
    Retrieve a Hugging Face model with device-aware caching.

    Behaviour:
    - Caches models by the pair (model_name, device or "auto").
    - Defaults to CUDA if available, otherwise CPU, when device is None.
    - Puts the model in evaluation mode and moves it onto the selected device.

    Args:
        model_name: model identifier.
        device: explicit device string such as "cuda" or "cpu"; if None, it is
            inferred from torch.cuda.is_available().

    Returns:
        A model instance on the requested device.
    """
    key = (model_name, device or "auto")
    m = _HF_MODEL_CACHE.get(key)
    if m is not None:
        return m
    import torch
    from transformers import AutoModel
    dev = device
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModel.from_pretrained(model_name)
    m.eval()
    m.to(dev)
    _HF_MODEL_CACHE[key] = m
    return m

def _hf_token_embeddings(text: str, model_name: str, device: str | None = None) -> tuple[np.ndarray, list[str]]:
    """
    Compute per-token embeddings for a text and filter out special tokens.

    Pipeline:
    - Tokenize the input text to obtain input_ids and attention_mask.
    - Run a forward pass to obtain last_hidden_state.
    - Remove padding and special tokens so that the output aligns with
      "content tokens" only.

    Returns:
        (embeddings, tokens):
            embeddings: numpy array of shape [L, D].
            tokens:     list of L string tokens corresponding to embeddings.

    Notes:
        If the model has a different structure (e.g. encoder attribute), a
        best-effort path is used to obtain a last_hidden_state-like tensor.
    """
    import torch
    tok = _get_hf_tokenizer(model_name)
    model = _get_hf_model(model_name, device=device)
    dev = next(model.parameters()).device
    enc = tok(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=getattr(tok, "model_max_length", 512)
    )
    input_ids = enc["input_ids"].to(dev)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(dev)
    with torch.no_grad():
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False, return_dict=True)
            hs = out.last_hidden_state[0]
        except Exception:
            if hasattr(model, "encoder"):
                out = model.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                hs = out.last_hidden_state[0]
            else:
                raise
    ids = input_ids[0].detach().cpu().tolist()
    mask = attention_mask[0].detach().cpu().numpy().astype(bool)
    try:
        spec_mask = tok.get_special_tokens_mask(ids, already_has_special_tokens=True)
        spec_mask = np.array(spec_mask, dtype=bool)
    except Exception:
        spec_mask = np.zeros(len(ids), dtype=bool)
    keep = mask & (~spec_mask)
    if keep.sum() < 1:
        keep = mask
    hs = hs.detach().cpu().numpy()
    hs = hs[keep]
    toks = tok.convert_ids_to_tokens([i for i, k in zip(ids, keep) if k])
    return hs, toks

def compute_tokenizer_embedding_cwt_delta(original_text: str, perturbed_text: str, model_name: str, *, scales: tuple[int, int] = (1, 64), device: str | None = None, wavelet: str = "cmor1.5-1.0") -> dict:
    """
    Compute CWT-based energy difference ΔE from tokenizer embeddings.

    Steps:
        1) Extract token embeddings for the original and perturbed texts and
           truncate both to the shared length L.
        2) Fit a shared PCA(1) on the concatenated embeddings and project each
           sequence onto the first principal component, yielding two 1D signals.
           If PCA fails, fall back to using the L2 norm of the embeddings.
        3) Compute a continuous wavelet transform (CWT) for each 1D signal and
           derive the energy E = |W|^2.
        4) Return the energy difference matrix ΔE = E_pert - E_orig together
           with token metadata and a simple mean-difference summary.

    Args:
        original_text: original input string.
        perturbed_text: perturbed variant of the original string.
        model_name: Hugging Face model identifier used for embeddings.
        scales: integer scale range (inclusive) used as CWT widths.
        device: optional device string ("cuda" / "cpu"); inferred if None.
        wavelet: wavelet name for pywt.cwt (default "cmor1.5-1.0"; falls back
            to scipy.signal.ricker if PyWavelets is unavailable).

    Returns:
        dict with keys:
        - "delta_energy": numpy array [S, T] containing ΔE.
        - "toks_o": list of original tokens (truncated to length L).
        - "toks_p": list of perturbed tokens (truncated to length L).
        - "mean_delta": float mean difference between projected signals.
    """
    emb_o, toks_o = _hf_token_embeddings(original_text, model_name, device=device)
    emb_p, toks_p = _hf_token_embeddings(perturbed_text, model_name, device=device)
    L = int(min(len(emb_o), len(emb_p)))
    if L < 2:
        return {"delta_energy": np.zeros((scales[1] - scales[0] + 1, max(1, L))), "toks_o": toks_o, "toks_p": toks_p, "mean_delta": 0.0}
    emb_o = emb_o[:L]
    emb_p = emb_p[:L]
    try:
        from sklearn.decomposition import PCA
        combined = np.vstack([emb_o, emb_p])
        pca = PCA(n_components=1)
        pca.fit(combined)
        sig_o = pca.transform(emb_o).reshape(-1)
        sig_p = pca.transform(emb_p).reshape(-1)
        if float(np.mean(sig_o)) < 0.0:
            sig_o = -sig_o
            sig_p = -sig_p
    except Exception:
        sig_o = np.linalg.norm(emb_o, axis=1)
        sig_p = np.linalg.norm(emb_p, axis=1)
    widths = np.arange(scales[0], scales[1] + 1)
    try:
        import pywt  # type: ignore
        cwt_o, _ = pywt.cwt(sig_o, widths, wavelet, sampling_period=1.0)
        cwt_p, _ = pywt.cwt(sig_p, widths, wavelet, sampling_period=1.0)
    except Exception:
        cwt_o = signal.cwt(sig_o, signal.ricker, widths)
        cwt_p = signal.cwt(sig_p, signal.ricker, widths)
    Eo = np.abs(cwt_o) ** 2
    Ep = np.abs(cwt_p) ** 2
    dE = Ep - Eo
    return {"delta_energy": dE, "toks_o": toks_o[:L], "toks_p": toks_p[:L], "mean_delta": float(np.mean(sig_p - sig_o))}

class TextWaveletAnalyzer:
    """
    Compute wavelet-derived features for (original, perturbed) text pairs.

    The analyzer attempts to use sentence-transformer token embeddings to construct a semantic
    trajectory. If the embedding backend is unavailable, it falls back to character-code signals
    to keep the pipeline robust.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Args:
            model_name: SentenceTransformer model identifier used for token embeddings.
        """
        self.model = None
        self.model_name = model_name
        
    def _load_model(self):
        """
        Lazily load the SentenceTransformer backend.

        This method keeps import-time dependencies optional: downstream users can still run the
        study without sentence-transformers installed, in which case wavelet analysis falls back.
        """
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"Failed to load SentenceTransformer: {e}")
                self.model = None

    def _get_embeddings(self, text):
        """
        Extract per-token embedding vectors for a text.

        Returns:
            ndarray of shape [tokens, dim] if available; otherwise None.
        """
        self._load_model()
        if self.model is None:
            return None
        try:
            # Tokenize and get embeddings via underlying transformer
            features = self.model.tokenize([text])
            out = self.model.forward(features)
            # token_embeddings: [batch, tokens, dim] -> [tokens, dim]
            tokens = out['token_embeddings'][0].detach().cpu().numpy()
            return tokens
        except Exception:
            return None

    def _get_shared_pca_signal(self, text_o, text_p):
        """
        Construct comparable 1D signals by fitting a shared PCA basis.

        Procedure:
        1) Extract token embeddings for both sequences.
        2) Fit PCA(n_components=1) on the concatenated embeddings.
        3) Project each sequence onto the first principal component to obtain 1D trajectories.

        Returns:
            sig_o, sig_p: 1D numpy arrays (unresampled) for the original and perturbed text.
        """
        emb_o = self._get_embeddings(text_o)
        emb_p = self._get_embeddings(text_p)
        
        # Fallback if embeddings fail
        if emb_o is None or emb_p is None:
            return self._char_code_signal(text_o), self._char_code_signal(text_p)
            
        try:
            from sklearn.decomposition import PCA
            combined = np.vstack([emb_o, emb_p])
            pca = PCA(n_components=1)
            pca.fit(combined)
            
            sig_o = pca.transform(emb_o).flatten()
            sig_p = pca.transform(emb_p).flatten()
            
            # Orient signal so mean is positive (heuristic for consistency)
            if np.mean(sig_o) < 0:
                sig_o = -sig_o
                sig_p = -sig_p
                
            return sig_o, sig_p
        except Exception:
             return self._char_code_signal(text_o), self._char_code_signal(text_p)

    def _sentence_vec(self, text):
        """
        Compute a single sentence embedding vector.

        Returns:
            1D numpy array if available; otherwise None.
        """
        self._load_model()
        if self.model is None:
            return None
        try:
            v = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=False)
            return v[0]
        except Exception:
            return None

    def token_similarity_series(self, original_text, perturbed_text):
        """
        Compute per-position cosine similarity between aligned token embeddings.

        The sequences are truncated to the shared minimum token length. This is a diagnostic
        signal (not a metric) that can be used to visualize local semantic drift.
        """
        self._load_model()
        if self.model is None:
            return None
        try:
            feats_o = self.model.tokenize([original_text])
            feats_p = self.model.tokenize([perturbed_text])
            out_o = self.model.forward(feats_o)
            out_p = self.model.forward(feats_p)
            to = out_o['token_embeddings'][0].detach().cpu().numpy()
            tp = out_p['token_embeddings'][0].detach().cpu().numpy()
            L = min(len(to), len(tp))
            if L < 1:
                return None
            to = to[:L]
            tp = tp[:L]
            num = np.sum(to * tp, axis=1)
            den = (np.linalg.norm(to, axis=1) * np.linalg.norm(tp, axis=1) + 1e-12)
            cos = num / den
            return cos
        except Exception:
            return None

    def _char_boundary_signal(self, text: str) -> np.ndarray:
        """
        Build a sparse boundary signal highlighting alphanumeric splits by whitespace.

        The returned signal has value 1.0 at indices where a whitespace character appears between
        two alphanumeric characters; otherwise 0.0. This is useful for capturing boundary-centric
        perturbations (e.g., zero-width spaces) even when embeddings are unavailable.
        """
        ws = {
            '\u0009', '\u000A', '\u000B', '\u000C', '\u000D',
            '\u0020', '\u0085', '\u00A0', '\u1680',
            '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', '\u200A',
            '\u2028', '\u2029', '\u202F', '\u205F', '\u3000', '\u200B'
        }
        n = len(text)
        sig = np.zeros(n, dtype=float)
        for i, ch in enumerate(text):
            if ch in ws:
                left = text[i-1] if i-1 >= 0 else ''
                right = text[i+1] if i+1 < n else ''
                if left.isalnum() and right.isalnum():
                    sig[i] = 1.0
        return sig

    def _char_code_signal(self, text: str) -> np.ndarray:
        """
        Convert text to a simple numeric signal using Unicode code points.

        This is a robust fallback when embedding-based signals cannot be computed.
        """
        return np.array([ord(c) for c in text], dtype=float)

    def text_to_signal(self, text):
        """
        Legacy: Converts text to a 1D signal using word embedding norms.
        """
        self._load_model()
        if self.model is None:
            return np.array([ord(c) for c in text], dtype=float)
            
        try:
            features = self.model.tokenize([text])
            out = self.model.forward(features)
            tokens = out['token_embeddings'][0].detach().cpu().numpy()
            signal_1d = np.linalg.norm(tokens, axis=1)
            if len(signal_1d) < 5:
                 return np.array([ord(c) for c in text], dtype=float)
            return signal_1d
        except Exception:
            return np.array([ord(c) for c in text], dtype=float)

    def compute_cross_similarity(self, original_text, perturbed_text):
        """
        Compute a token-token cross-similarity matrix (cosine similarity).

        Returns:
            sim_matrix: ndarray [L_orig, L_pert] with cosine similarities, or None if unavailable.
            toks_o: Token strings for the original text.
            toks_p: Token strings for the perturbed text.
        """
        self._load_model()
        if self.model is None:
            return None, [], []
        
        try:
            # Get token embeddings
            feats_o = self.model.tokenize([original_text])
            feats_p = self.model.tokenize([perturbed_text])
            out_o = self.model.forward(feats_o)
            out_p = self.model.forward(feats_p)
            
            # Embeddings: [N, D]
            emb_o = out_o['token_embeddings'][0].detach().cpu().numpy()
            emb_p = out_p['token_embeddings'][0].detach().cpu().numpy()
            
            # Normalize for cosine similarity
            norm_o = np.linalg.norm(emb_o, axis=1, keepdims=True) + 1e-12
            norm_p = np.linalg.norm(emb_p, axis=1, keepdims=True) + 1e-12
            emb_o_n = emb_o / norm_o
            emb_p_n = emb_p / norm_p
            
            # Cross-Similarity Matrix: [Lx, Ly]
            sim_matrix = np.dot(emb_o_n, emb_p_n.T)
            
            # Get raw tokens for axis labels
            tokenizer = self.model.tokenizer
            ids_o = feats_o['input_ids'][0]
            ids_p = feats_p['input_ids'][0]
            toks_o = tokenizer.convert_ids_to_tokens(ids_o)
            toks_p = tokenizer.convert_ids_to_tokens(ids_p)
            
            return sim_matrix, toks_o, toks_p
            
        except Exception:
            return None, [], []

    def compute_pca_trajectory(self, original_text, perturbed_text):
        """
        Compute a shared 2D PCA projection of token embeddings for visualization.

        Returns:
            coords_o: ndarray [N, 2] for original tokens, or None if unavailable.
            coords_p: ndarray [M, 2] for perturbed tokens, or None if unavailable.
        """
        self._load_model()
        if self.model is None:
            return None, None
            
        try:
            from sklearn.decomposition import PCA
            
            feats_o = self.model.tokenize([original_text])
            feats_p = self.model.tokenize([perturbed_text])
            out_o = self.model.forward(feats_o)
            out_p = self.model.forward(feats_p)
            
            emb_o = out_o['token_embeddings'][0].detach().cpu().numpy()
            emb_p = out_p['token_embeddings'][0].detach().cpu().numpy()
            
            combined = np.vstack([emb_o, emb_p])
            
            pca = PCA(n_components=2)
            pca.fit(combined)
            
            coords_o = pca.transform(emb_o)
            coords_p = pca.transform(emb_p)
            
            return coords_o, coords_p
            
        except Exception:
            return None, None

    def compute_wavelet_features(self, original_text, perturbed_text):
        """
        Compute scalar wavelet features for batch storage.

        This is a lightweight wrapper around compute_pair_wavelets() that summarizes the
        differential energy and a high-frequency energy proxy into scalar columns.
        """
        try:
            res = self.compute_pair_wavelets(original_text, perturbed_text)
            
            # Use diff energy
            if res["energy_scalogram_diff"] is not None:
                total_energy = np.sum(np.abs(res["energy_scalogram_diff"]))
            else:
                total_energy = 0.0
                
            # HF Energy
            if res["energy_hf_time"] is not None:
                hf_energy = np.sum(res["energy_hf_time"])
            else:
                hf_energy = 0.0

            return {
                "wavelet_energy": total_energy,
                "wavelet_high_freq_energy": hf_energy,
                "wavelet_snr": res["snr"],
                "cwt_matrix": res["energy_scalogram_diff"] 
            }
        except Exception as e:
            return {
                "wavelet_energy": 0.0,
                "wavelet_high_freq_energy": 0.0,
                "wavelet_snr": 0.0,
                "cwt_matrix": None
            }

    def _resample(self, sig: np.ndarray, target_len: int = 128) -> np.ndarray:
        """
        Resamples a 1D signal to a fixed target length using linear interpolation.
        Preserves the range and general shape.
        """
        if sig is None or len(sig) == 0:
            return np.zeros(target_len)
        if len(sig) == target_len:
            return sig
        x_old = np.linspace(0, 1, len(sig))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, sig)

    def compute_pair_wavelets(self, original_text, perturbed_text):
        """
        Compute full wavelet artifacts for a text pair (signals, scalograms, and summaries).

        Methodology (Continuous Wavelet Transform):
        We use CWT rather than Discrete Wavelet Transform (DWT) or Fourier Transform (DFT) because:
        1. Non-Stationarity: Text perturbations are transient events (e.g., a single typo). 
           DFT smears this across all frequencies; CWT preserves time localization.
        2. Redundancy: CWT provides a dense, shift-invariant representation (scalogram) ideal 
           for visualization and feature extraction, unlike the sparse, decimated DWT.
        3. Kernel: We use the Complex Morlet wavelet (`cmor`) because its Gaussian envelope 
           minimizes spectral leakage, providing the optimal trade-off between time and 
           frequency resolution (Heisenberg uncertainty).

        Pipeline:
        1) Signal construction: shared PCA projection.
        2) Resampling: unify time axis (0..1) to length 128.
        3) CWT: compute time–scale energy $E(t,s) = |W(t,s)|^2$.
        4) Differential Analysis: $\Delta E = E_{pert} - E_{orig}$.

        Returns:
            A dictionary containing signals, scalograms, differential energy, and summary series.
        """
        TARGET_W = 128
        
        # 1. 1D Signal Construction (PCA Projection)
        sig_o_raw, sig_p_raw = self._get_shared_pca_signal(original_text, perturbed_text)
        
        # 2. Resampling (Unified Time Axis 0..1)
        sig_o = self._resample(sig_o_raw, TARGET_W)
        sig_p = self._resample(sig_p_raw, TARGET_W)
        
        # 3. Wavelet Transform (CWT)
        # Using Morlet wavelet as recommended
        widths = np.arange(1, 65) # Scale 1..64
        try:
            import pywt # type: ignore
            # Use complex morlet 'cmor1.5-1.0' for standard Morlet-like behavior in pywt
            # Or 'morl' (real) if we want simple energy
            # User suggested "Morlet or Daubechies 4". cmor is better for scalograms (phase+amp).
            # But for simple energy, complex magnitude is good.
            cwt_o, _ = pywt.cwt(sig_o, widths, 'cmor1.5-1.0')
            cwt_p, _ = pywt.cwt(sig_p, widths, 'cmor1.5-1.0')
        except Exception:
            # Fallback
            cwt_o = signal.cwt(sig_o, signal.ricker, widths)
            cwt_p = signal.cwt(sig_p, signal.ricker, widths)

        # 4. Energy & Difference
        energy_o = np.abs(cwt_o) ** 2
        energy_p = np.abs(cwt_p) ** 2
        
        # Z-score normalization (per scalogram) to highlight structure over absolute magnitude
        # This makes the comparison fair if overall amplitude shifts
        def _znorm(mat):
            mu = np.mean(mat)
            sd = np.std(mat)
            return (mat - mu) / (sd + 1e-12)
            
        # Optional: Normalize before diff (z-score per scalogram).
        # We keep normalized versions for potential visualization use.
        energy_o_z = _znorm(energy_o)
        energy_p_z = _znorm(energy_p)
        
        # Difference Spectrum: W_perturb - W_orig
        # Using normalized versions highlights *pattern* changes
        # Using raw versions highlights *intensity* changes
        # Let's provide raw difference for physical interpretation, but z-score individual plots
        energy_diff = energy_p - energy_o
        
        # 5. Metrics
        E_time = np.sum(np.abs(energy_diff), axis=0)
        E_scale_diff = np.sum(energy_diff, axis=1) # Signed difference per scale
        
        # High-Freq Energy (First 8 scales)
        hf_band = 8
        hf_energy_profile = np.sum(np.abs(energy_diff[:hf_band]), axis=0)
        
        # SNR
        mse = np.mean((sig_p_raw - np.pad(sig_o_raw, (0, max(0, len(sig_p_raw)-len(sig_o_raw)))))**2) if len(sig_p_raw) >= len(sig_o_raw) else np.mean((sig_o_raw - np.pad(sig_p_raw, (0, len(sig_o_raw)-len(sig_p_raw))))**2)
        peak = np.max(np.abs(sig_o_raw)) + 1e-12
        psnr = 10.0 * np.log10(peak**2 / (mse + 1e-12))

        return {
            "cwt_original": cwt_o,
            "cwt_perturbed": cwt_p,
            "energy_original": energy_o,
            "energy_perturbed": energy_p,
            "energy_scalogram_diff": energy_diff,
            "energy_time": E_time,
            "energy_hf_time": hf_energy_profile,
            "energy_scale_diff": E_scale_diff,
            "signals": (sig_o, sig_p), # Resampled signals for plotting
            "raw_signals": (sig_o_raw, sig_p_raw),
            "snr": psnr,
            "composite_cwt": energy_diff # Alias for compatibility
        }
