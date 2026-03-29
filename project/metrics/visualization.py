"""
Visualization Metrics (Perceptual and Layout Stability).

Scientific Motivation:
This module quantifies the visual impact of textual perturbations.
Some perturbations (like homoglyph attacks or whitespace injection) may be semantically
and syntactically disruptive to models but visually imperceptible to humans.
Conversely, layout shifts can break reading flow even if content is preserved.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
import platform

def get_font(size=24):
    system = platform.system()
    try:
        if system == "Windows":
            return ImageFont.truetype("arial.ttf", size)
        elif system == "Darwin": # MacOS
            return ImageFont.truetype("Arial.ttf", size)
        else: # Linux
            return ImageFont.truetype("DejaVuSans.ttf", size)
    except IOError:
        return ImageFont.load_default()

def render_text_to_array(text, font_size=32, image_size=(1024, 100)):
    font = get_font(font_size)
    image = Image.new("L", image_size, "white")
    draw = ImageDraw.Draw(image)
    
    # Render at top-left with padding
    draw.text((10, 10), text, font=font, fill="black")
    
    return np.array(image)

def rendered_ssim_distance(s: str, sp: str) -> float:
    """
    Calculates the Structural Similarity Distance (SSIMD).

    Definition:
        Render text as images I_x and I_x~.
        SSIMD(x, x~) = 1 - SSIM(I_x, I_x~).

    Role:
        Quantifies visual similarity.
        Character changes, spacing, or punctuation shifts that look similar lead to low SSIMD.
        Range: [0, 2], typically [0, 1].

    Args:
        s: Original text sequence x.
        sp: Perturbed text sequence x~.

    Returns:
        float: The visual distance score.
    """
    # Create a canvas large enough for both strings
    width = max(len(s), len(sp)) * 20 + 50
    size = (width, 100)
    
    img_s = render_text_to_array(s, image_size=size)
    img_sp = render_text_to_array(sp, image_size=size)
    
    # Compute SSIM
    # data_range is max - min (255 - 0 = 255)
    score, _ = ssim(img_s, img_sp, full=True, data_range=255)
    
    return 1.0 - score

def glyph_layout_displacement(s: str, sp: str) -> float:
    """
    Calculates the Glyph Layout Displacement (GD).

    Definition:
        GD(x, x~) approximates the average physical displacement of glyphs.
        Ideally: (1/|P|) * sum ||u(p)||_2 / L.
        Approximation: |Width(x) - Width(x~)| / Font_Size.

    Role:
        Captures how far glyph shapes move in the image plane due to character edits.
        Complements SSIMD: SSIMD detects structural changes, GD measures geometric displacement.
        Range: [0, +inf).

    Args:
        s: Original text sequence x.
        sp: Perturbed text sequence x~.

    Returns:
        float: The normalized layout displacement.
    """
    font_size = 24
    font = get_font(font_size)
    
    # Get text length in pixels
    # Pillow 9.2.0+ uses getlength
    try:
        len_s = font.getlength(s)
        len_sp = font.getlength(sp)
    except AttributeError:
        len_s = font.getsize(s)[0]
        len_sp = font.getsize(sp)[0]
        
    diff = abs(len_s - len_sp)
    
    # Normalize by font size as a rough proxy for "displacement" magnitude relative to line height
    return diff / float(font_size)

def spatial_dispersion_salience_score(s: str, sp: str) -> float:
    """
    Calculates the Spatial Dispersion Salience Score (SDSS).

    Definition:
        SDSS(x, x~) measures the shift in visual center of mass and dispersion variance.
        SDSS = (|mu_x - mu_x~| + |sigma_x - sigma_x~|) / Z.

    Role:
        Measures where human visual attention would move after perturbation.
        Character changes that alter focal points cause large SDSS.
        Range: Normalized to approx [0, 1].

    Args:
        s: Original text sequence x.
        sp: Perturbed text sequence x~.

    Returns:
        float: The spatial dispersion score.
    """
    def weights_for(x: str):
        # Heuristic weights: space is low attention, symbols high, uppercase high
        w = []
        for ch in x:
            if ch == ' ':
                w.append(0.5)
            elif ch.isspace():
                w.append(1.5)
            elif ch.isupper():
                w.append(1.1)
            elif ch.isdigit():
                w.append(0.9)
            else:
                w.append(1.0)
        return w
        
    def center_of_mass(weights):
        if not weights:
            return 0.0, 0.0, 0.0
        pos = []
        cum = 0.0
        for wt in weights:
            cum += wt
            pos.append(cum)
        total_w = sum(weights)
        com = sum(wt * p for wt, p in zip(weights, pos)) / total_w if total_w > 0 else 0.0
        var = sum(((p - com) ** 2) for p in pos) / len(pos) if pos else 0.0
        return com, var, pos[-1] if pos else 0.0
        
    w_o = weights_for(s)
    w_d = weights_for(sp)
    
    com_o, var_o, width_o = center_of_mass(w_o)
    com_d, var_d, width_d = center_of_mass(w_d)
    
    total_w = max(width_o, width_d, 1e-6)
    
    # Center of mass shift (normalized by total width)
    com_diff = abs(com_d - com_o) / total_w
    
    # Variance shift (normalized by original variance)
    var_diff = abs(var_d - var_o) / (var_o + 1e-6)
    
    return 0.5 * com_diff + 0.5 * var_diff
