import numpy as np


def centrality_to_hex(value: float, vmin: float = 0.0, vmax: float = 1.0) -> str:
    """Map a centrality value [vmin, vmax] to a red-yellow-green hex color."""
    t = np.clip((value - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    r = int(255 * min(1.0, 2 * t))
    g = int(255 * min(1.0, 2 * (1 - t)))
    b = 30
    return f"#{r:02x}{g:02x}{b:02x}"


def resilience_color(score: float) -> str:
    """Return a hex color for a 0–100 resilience score."""
    if score >= 75:
        return "#2ecc71"
    if score >= 50:
        return "#f39c12"
    return "#e74c3c"
