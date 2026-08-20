"""
Shared configuration: palette, plotting style, paths.

Palette follows the categorical/sequential/status conventions from the
project's data-viz style guide (fixed hue order, single-hue sequential
ramp, reserved status colors). Values below are hex strings, matplotlib-ready.
"""
import os
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- paths ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "figures")

# ---------------------------------------------------------------- palette --
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}
INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "baseline": "#c3c2b7",
    "surface": "#fcfcfb",
}

SEGMENT_ORDER = ["Enterprise", "Mid-Market", "SMB", "Occasional"]
SEGMENT_COLOR = dict(zip(SEGMENT_ORDER, CATEGORICAL[:4]))

RANDOM_SEED = 42


def set_style():
    plt.rcParams.update({
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],
        "axes.edgecolor": INK["baseline"],
        "axes.labelcolor": INK["secondary"],
        "text.color": INK["primary"],
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "grid.color": INK["grid"],
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "figure.dpi": 110,
    })
