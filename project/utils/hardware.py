"""
Hardware and accelerator diagnostics (GPU / CUDA / related libraries).

This module provides a small utility layer to:
- Detect whether NVIDIA GPUs and CUDA are available.
- Summarize basic device information and supporting libraries (PyTorch, Numba, XGBoost).
- Format the collected information into a human-readable string for logging.
"""
import subprocess
import shutil

def get_gpu_diagnostics():
    """
    Collect hardware and accelerator availability information.

    Returns:
        dict with the following keys:
        - nvidia_smi: whether the nvidia-smi binary is available.
        - nvidia_smi_output: first line of nvidia-smi -L output (device summary), if any.
        - torch_cuda: whether torch.cuda.is_available() is True.
        - torch_devices: list of device names reported by PyTorch.
        - numba_cuda: whether Numba detects a CUDA-capable device.
        - xgb_version: installed XGBoost version string, or None.
        - xgb_gpu_param_supported: whether modern XGBoost GPU parameters are supported (v2+).
    """
    info = {
        "nvidia_smi": False,
        "nvidia_smi_output": None,
        "torch_cuda": False,
        "torch_devices": [],
        "numba_cuda": False,
        "xgb_version": None,
        "xgb_gpu_param_supported": False
    }
    try:
        path = shutil.which("nvidia-smi")
        if path:
            r = subprocess.run([path, "-L"], capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                info["nvidia_smi"] = True
                info["nvidia_smi_output"] = r.stdout.strip()
    except Exception:
        pass
    try:
        import torch
        info["torch_cuda"] = bool(torch.cuda.is_available())
        if info["torch_cuda"]:
            c = torch.cuda.device_count()
            info["torch_devices"] = [torch.cuda.get_device_name(i) for i in range(c)]
    except Exception:
        pass
    try:
        from numba import cuda as _cuda
        info["numba_cuda"] = bool(_cuda.is_available())
    except Exception:
        pass
    try:
        import xgboost as xgb
        info["xgb_version"] = getattr(xgb, "__version__", None)
        try:
            ver = info["xgb_version"] or "0"
            parts = ver.split(".")
            major = int(parts[0])
            info["xgb_gpu_param_supported"] = major >= 2
        except Exception:
            info["xgb_gpu_param_supported"] = False
    except Exception:
        pass
    return info

def format_gpu_diagnostics(info):
    """
    Format the diagnostics dictionary into a human-readable multi-line string.

    Args:
        info: dictionary returned by get_gpu_diagnostics().

    Returns:
        str: summary text suitable for console or log output.
    """
    lines = []
    lines.append(f"NVIDIA-SMI: {'OK' if info.get('nvidia_smi') else 'Not found'}")
    if info.get("nvidia_smi_output"):
        lines.append(info["nvidia_smi_output"].splitlines()[0])
    lines.append(f"PyTorch CUDA: {'Available' if info.get('torch_cuda') else 'Unavailable'}")
    if info.get("torch_devices"):
        lines.append("Devices: " + ", ".join(info["torch_devices"]))
    lines.append(f"Numba CUDA: {'Available' if info.get('numba_cuda') else 'Unavailable'}")
    xv = info.get("xgb_version") or "Unknown"
    lines.append(f"XGBoost: {xv} (GPU param {'device=cuda' if info.get('xgb_gpu_param_supported') else 'gpu_hist'} supported)")
    return "\n".join(lines)
