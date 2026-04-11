from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_chart(
    data: dict[str, dict[str, float]],
    title: str,
    filepath: Path,
    lower_is_better: set[str] | None = None,
) -> None:
    """Generate and save a grouped bar chart for metrics.

    Args:
        data:    {method_name: {metric_name: value}}.
        title:   Chart title.
        filepath: Output PNG path.
        lower_is_better: Set of metric names where lower = better (shown in orange).
    """
    if lower_is_better is None:
        lower_is_better = set()

    methods = list(data.keys())
    metrics = list(next(iter(data.values())).keys())
    n_methods = len(methods)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors_higher = plt.cm.Blues(np.linspace(0.45, 0.85, n_methods))
    colors_lower  = plt.cm.Oranges(np.linspace(0.45, 0.85, n_methods))

    for ax, metric in zip(axes, metrics):
        vals = [data[m].get(metric, 0.0) for m in methods]
        is_lower = metric in lower_is_better
        colors = colors_lower if is_lower else colors_higher
        bars = ax.bar(range(n_methods), vals, color=colors)
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
        direction = "↓" if is_lower else "↑"
        ax.set_title(f"{metric} ({direction})", fontsize=11, fontweight="bold")
        ax.set_ylabel(metric)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(filepath), dpi=150)
    plt.close(fig)
