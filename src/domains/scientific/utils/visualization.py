"""
Visualization Utilities — Metric Charts and Distribution Plots.

Provides plotting functions for comparing model performance, analysing
confidence distributions, and visualizing retrieval precision.  All
functions use matplotlib with Noto Sans SC + DejaVu Sans font fallback
for compatibility across environments (Kaggle notebooks, local
development, etc.).

Functions
---------
* :func:`plot_metrics_comparison` — Grouped bar chart comparing
  baseline vs. our system across multiple metrics.
* :func:`plot_confidence_distribution` — Histogram of confidence
  scores from the self-check pipeline.
* :func:`plot_retrieval_precision` — Precision@k line chart showing
  retrieval accuracy at different cut-off values.

Example:
    >>> from src.domains.scientific.utils.visualization import plot_metrics_comparison
    >>> baseline = {"bleu4": 0.30, "rouge_l": 0.50, "anls": 0.65, "f1": 0.52}
    >>> ours = {"bleu4": 0.45, "rouge_l": 0.62, "anls": 0.78, "f1": 0.65}
    >>> plot_metrics_comparison(baseline, ours, Path("output/comparison.png"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.shared.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Matplotlib setup with font compatibility
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# Try to configure fonts for CJK + Latin compatibility
_FONT_CONFIGURED = False

if _HAS_MPL:
    try:
        # Try Noto Sans SC for CJK characters
        noto_fonts = [
            f.name for f in fm.fontManager.ttflist
            if "Noto Sans" in f.name
        ]
        if noto_fonts:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"] + plt.rcParams.get("font.sans-serif", [])
            _FONT_CONFIGURED = True
            logger.debug("Configured Noto Sans SC + DejaVu Sans fonts.")
        else:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["DejaVu Sans"] + plt.rcParams.get("font.sans-serif", [])
            _FONT_CONFIGURED = True
            logger.debug("Noto Sans SC not found — using DejaVu Sans.")
    except Exception as exc:
        logger.warning("Font configuration failed: %s — using defaults.", exc)

    # Common style settings
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

_COLORS = {
    "baseline": "#94a3b8",  # Slate gray
    "ours": "#3b82f6",      # Blue
    "accent": "#10b981",    # Green
    "warning": "#f59e0b",   # Amber
    "danger": "#ef4444",    # Red
}


# ---------------------------------------------------------------------------
# plot_metrics_comparison
# ---------------------------------------------------------------------------

def plot_metrics_comparison(
    baseline_metrics: Dict[str, float],
    our_metrics: Dict[str, float],
    output_path: Path,
    title: str = "Metrics Comparison: Baseline vs. Ours",
) -> None:
    """Plot a grouped bar chart comparing baseline vs. our metrics.

    Creates a bar chart with pairs of bars (baseline and ours) for
    each metric.  Our bars are highlighted with the accent color and
    include value labels.

    Args:
        baseline_metrics: Dictionary of baseline metric scores,
            e.g. ``{"bleu4": 0.30, "rouge_l": 0.50}``.
        our_metrics: Dictionary of our system's metric scores,
            with the same keys as *baseline_metrics*.
        output_path: Path to save the chart image.  The parent
            directory is created if it doesn't exist.
        title: Chart title string.

    Example:
        >>> baseline = {"bleu4": 0.30, "rouge_l": 0.50}
        >>> ours = {"bleu4": 0.45, "rouge_l": 0.62}
        >>> plot_metrics_comparison(baseline, ours, Path("comparison.png"))
    """
    if not _HAS_MPL:
        logger.warning("matplotlib not available — skipping plot.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use only keys present in both dictionaries
    metrics = sorted(set(baseline_metrics.keys()) & set(our_metrics.keys()))

    if not metrics:
        logger.warning("No common metrics to compare.")
        return

    baseline_vals = [baseline_metrics[m] for m in metrics]
    our_vals = [our_metrics[m] for m in metrics]

    # Capitalize metric names for display
    labels = [m.upper().replace("_", "-") for m in metrics]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.5), 5))

    bars_baseline = ax.bar(
        [xi - width / 2 for xi in x],
        baseline_vals,
        width,
        label="Baseline",
        color=_COLORS["baseline"],
        edgecolor="white",
        linewidth=0.5,
    )

    bars_ours = ax.bar(
        [xi + width / 2 for xi in x],
        our_vals,
        width,
        label="Ours",
        color=_COLORS["ours"],
        edgecolor="white",
        linewidth=0.5,
    )

    # Add value labels on bars
    for bar in bars_baseline:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=_COLORS["baseline"],
        )

    for bar in bars_ours:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=_COLORS["ours"],
        )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, max(max(baseline_vals), max(our_vals)) * 1.15)

    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)

    logger.info("Metrics comparison chart saved to: %s", output_path)


# ---------------------------------------------------------------------------
# plot_confidence_distribution
# ---------------------------------------------------------------------------

def plot_confidence_distribution(
    confidences: List[float],
    output_path: Path,
    title: str = "Confidence Score Distribution",
    threshold: float = 0.6,
) -> None:
    """Plot a histogram of confidence scores from the self-check.

    Visualizes the distribution of confidence scores with a vertical
    line marking the confidence threshold.  Scores below the
    threshold are colored differently to highlight answers that would
    be retried.

    Args:
        confidences: List of confidence scores, each in ``[0, 1]``.
        output_path: Path to save the chart image.
        title: Chart title string.
        threshold: Confidence threshold to mark on the chart.
            Defaults to 0.6.

    Example:
        >>> scores = [0.3, 0.5, 0.7, 0.8, 0.9, 0.6, 0.4, 0.85]
        >>> plot_confidence_distribution(scores, Path("confidence.png"))
    """
    if not _HAS_MPL:
        logger.warning("matplotlib not available — skipping plot.")
        return

    if not confidences:
        logger.warning("Empty confidence list — nothing to plot.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Separate into above/below threshold for coloring
    below = [c for c in confidences if c < threshold]
    above = [c for c in confidences if c >= threshold]

    bins = 20
    bin_range = (0.0, 1.0)

    ax.hist(
        above,
        bins=bins,
        range=bin_range,
        color=_COLORS["accent"],
        alpha=0.8,
        label=f"Pass (≥ {threshold:.1f})",
        edgecolor="white",
        linewidth=0.5,
    )

    ax.hist(
        below,
        bins=bins,
        range=bin_range,
        color=_COLORS["danger"],
        alpha=0.8,
        label=f"Fail (< {threshold:.1f})",
        edgecolor="white",
        linewidth=0.5,
    )

    # Threshold line
    ax.axvline(
        x=threshold,
        color=_COLORS["warning"],
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.1f}",
    )

    # Statistics annotation
    mean_conf = sum(confidences) / len(confidences)
    pass_rate = len(above) / len(confidences) * 100

    stats_text = (
        f"Mean: {mean_conf:.3f}\n"
        f"Pass rate: {pass_rate:.1f}%\n"
        f"N = {len(confidences)}"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fancybox=True, shadow=True)

    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)

    logger.info(
        "Confidence distribution chart saved to: %s "
        "(mean=%.3f, pass_rate=%.1f%%)",
        output_path,
        mean_conf,
        pass_rate,
    )


# ---------------------------------------------------------------------------
# plot_retrieval_precision
# ---------------------------------------------------------------------------

def plot_retrieval_precision(
    precision_data: Dict[str, List[float]],
    output_path: Path,
    title: str = "Retrieval Precision@k",
    k_values: Optional[List[int]] = None,
) -> None:
    """Plot precision@k chart for retrieval evaluation.

    Creates a line chart showing precision at different cut-off
    values (k) for one or more retrieval methods.  Each method
    is plotted as a separate line with markers.

    Args:
        precision_data: Dictionary mapping method names to lists of
            precision values.  For example::

                {
                    "ColPali": [1.0, 0.9, 0.8, 0.7, 0.6],
                    "SciNCL": [0.8, 0.7, 0.65, 0.6, 0.55],
                    "Fusion": [1.0, 0.95, 0.85, 0.75, 0.7],
                }

        output_path: Path to save the chart image.
        title: Chart title string.
        k_values: List of k values for the x-axis.  If ``None``,
            defaults to ``[1, 3, 5, 10, 20]``.  The length must
            match the precision lists in *precision_data*.

    Example:
        >>> data = {"Fusion": [1.0, 0.9, 0.8], "ColPali": [0.8, 0.7, 0.6]}
        >>> plot_retrieval_precision(data, Path("precision.png"), k_values=[1, 3, 5])
    """
    if not _HAS_MPL:
        logger.warning("matplotlib not available — skipping plot.")
        return

    if not precision_data:
        logger.warning("Empty precision data — nothing to plot.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    fig, ax = plt.subplots(figsize=(8, 5))

    line_colors = [_COLORS["ours"], _COLORS["accent"], _COLORS["baseline"], _COLORS["warning"]]
    markers = ["o", "s", "^", "D", "v"]

    for idx, (method_name, precisions) in enumerate(precision_data.items()):
        # Adjust k_values length to match precision list
        ks = k_values[:len(precisions)]

        if len(ks) < len(precisions):
            ks = list(range(1, len(precisions) + 1))

        color = line_colors[idx % len(line_colors)]
        marker = markers[idx % len(markers)]

        ax.plot(
            ks,
            precisions[:len(ks)],
            marker=marker,
            label=method_name,
            color=color,
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("k (top-k results)")
    ax.set_ylabel("Precision@k")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)

    logger.info("Retrieval precision chart saved to: %s", output_path)
