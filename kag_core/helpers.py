"""
Helper functions for ViT Jacobian analysis.
Reusable utilities for loading and analyzing KA geometry metrics.

IMPORTANT: Always check all THREE KA metrics together:
1. Rotation Ratio - Anisotropy (directional dependence)
2. KL Divergence - Non-uniformity of distribution
3. Participation Ratio (PR) - Concentration of column norms
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Data Loading
# =============================================================================


def load_metrics(path: str | Path) -> dict[str, Any]:
    """Load metrics from JSON file."""
    with open(path) as f:
        return json.load(f)


def parse_tensor_string(tensor_str: str) -> list[float]:
    """Parse a PyTorch tensor string representation to list of floats.

    Handles truncated tensor strings like:
    "tensor([0.0014, 0.0014, ...])"
    """
    if not isinstance(tensor_str, str):
        return list(tensor_str) if hasattr(tensor_str, "__iter__") else [tensor_str]

    # Remove tensor wrapper
    cleaned = tensor_str.replace("tensor([", "").replace("])", "").replace("\n", "")

    nums = []
    for x in cleaned.split(","):
        x = x.strip()
        if x and "..." not in x:
            try:
                nums.append(float(x))
            except ValueError:
                pass
    return nums


def load_run_data(run_dir: str | Path) -> dict[str, dict]:
    """Load all available data from a run directory.

    Returns dict with 'init', 'full_init', and 'timeline' keys.
    """
    run_dir = Path(run_dir)
    data = {}

    # Full precision init (metrics.json)
    metrics_json = run_dir / "metrics.json"
    if metrics_json.exists():
        data["full_init"] = load_metrics(metrics_json)

    # Truncated init (metrics_init.json)
    init_json = run_dir / "metrics_init.json"
    if init_json.exists():
        data["init"] = load_metrics(init_json)

    # Timeline snapshots
    timeline_dir = run_dir / "timeline"
    if timeline_dir.exists():
        data["timeline"] = {}
        for f in sorted(timeline_dir.glob("epoch_*.json")):
            epoch = int(f.stem.split("_")[1])
            data["timeline"][epoch] = load_metrics(f)

    return data


def list_runs(results_dir: str = "vit_jacobian_results") -> list[Path]:
    """List all run directories."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return []
    return sorted([d for d in results_path.iterdir() if d.is_dir()])


# =============================================================================
# Metric Extraction
# =============================================================================


def get_metric_values(data: dict, block_key: str, k: str, metric: str) -> np.ndarray:
    """Extract metric values from data structure.

    Args:
        data: Loaded metrics dict
        block_key: e.g., 'block0_layer0'
        k: '1' or '2' for k-value
        metric: 'rotation_ratios', 'participation_ratios', 'kl_divergence', etc.

    Returns:
        numpy array of values
    """
    raw = data[block_key][k][metric]

    if isinstance(raw, list):
        return np.array(raw)
    elif isinstance(raw, str):
        return np.array(parse_tensor_string(raw))
    else:
        return np.array([raw])


def get_metric_mean(data: dict, block_key: str, k: str, metric: str) -> float:
    """Get mean of a metric."""
    values = get_metric_values(data, block_key, k, metric)
    return float(np.mean(values))


def get_rotation_ratio(data: dict, block_key: str, k: str = "1") -> float:
    """Get mean rotation ratio."""
    return get_metric_mean(data, block_key, k, "rotation_ratios")


def get_pr(data: dict, block_key: str, k: str = "1") -> float:
    """Get mean participation ratio (L2/L1)."""
    try:
        return get_metric_mean(data, block_key, k, "participation_ratios")
    except KeyError:
        return get_metric_mean(data, block_key, k, "pr_values")


def get_kl(data: dict, block_key: str, k: str = "1") -> float:
    """Get mean KL divergence."""
    return get_metric_mean(data, block_key, k, "kl_divergence")


def get_column_std(data: dict, block_key: str, k: str = "1") -> float:
    """Get mean column norm std."""
    return get_metric_mean(data, block_key, k, "std_values")


# =============================================================================
# Comparison Utilities
# =============================================================================


def compare_metrics(
    init_data: dict,
    trained_data: dict,
    blocks: list[int] = None,
    layer: int = 0,
    k: str = "1",
    metric_fn=None,
    metric_name: str = "Metric",
) -> dict[str, Any]:
    """Compare metrics between init and trained states.

    Returns dict with per-block and average changes.
    """
    if metric_fn is None:
        metric_fn = get_rotation_ratio

    if blocks is None:
        blocks = list(range(12))

    results = {"blocks": {}, "init_values": [], "trained_values": [], "changes_pct": []}

    for block_idx in blocks:
        key = f"block{block_idx}_layer{layer}"

        init_val = metric_fn(init_data, key, k)
        trained_val = metric_fn(trained_data, key, k)
        change_pct = (trained_val / init_val - 1) * 100 if init_val != 0 else 0

        results["blocks"][block_idx] = {
            "init": init_val,
            "trained": trained_val,
            "change_pct": change_pct,
        }
        results["init_values"].append(init_val)
        results["trained_values"].append(trained_val)
        results["changes_pct"].append(change_pct)

    results["avg_init"] = np.mean(results["init_values"])
    results["avg_trained"] = np.mean(results["trained_values"])
    results["avg_change_pct"] = np.mean(results["changes_pct"])

    return results


def print_comparison_table(
    results: dict[str, Any], metric_name: str = "Metric", precision: int = 6
):
    """Print a formatted comparison table."""
    print(f"{'Block':<8} {'Init':>{precision + 6}} {'Trained':>{precision + 6}} {'Change':>12}")
    print("-" * (8 + (precision + 6) * 2 + 12 + 6))

    for block_idx, vals in results["blocks"].items():
        print(
            f"Block {block_idx:<3} {vals['init']:>{precision + 6}.{precision}f} "
            f"{vals['trained']:>{precision + 6}.{precision}f} {vals['change_pct']:>+11.1f}%"
        )

    print("-" * (8 + (precision + 6) * 2 + 12 + 6))
    print(
        f"{'AVERAGE':<8} {results['avg_init']:>{precision + 6}.{precision}f} "
        f"{results['avg_trained']:>{precision + 6}.{precision}f} {results['avg_change_pct']:>+11.1f}%"
    )


# =============================================================================
# Summary Statistics
# =============================================================================


def summarize_run(run_data: dict, epochs: list[int] = None) -> dict[str, Any]:
    """Generate summary statistics for a run."""
    summary = {}

    if "full_init" in run_data:
        init = run_data["full_init"]
    elif "init" in run_data:
        init = run_data["init"]
    else:
        return summary

    if epochs is None and "timeline" in run_data:
        epochs = sorted(run_data["timeline"].keys())

    # Init stats
    summary["init"] = {
        "rotation_ratio_k1": np.mean(
            [get_rotation_ratio(init, f"block{i}_layer0", "1") for i in range(12)]
        ),
        "rotation_ratio_k2": np.mean(
            [get_rotation_ratio(init, f"block{i}_layer0", "2") for i in range(12)]
        ),
        "kl_k1": np.mean([get_kl(init, f"block{i}_layer0", "1") for i in range(12)]),
        "pr_k1": np.mean([get_pr(init, f"block{i}_layer0", "1") for i in range(12)]),
    }

    # Timeline stats
    if "timeline" in run_data:
        for epoch in epochs:
            if epoch in run_data["timeline"]:
                trained = run_data["timeline"][epoch]
                summary[f"epoch_{epoch}"] = {
                    "rotation_ratio_k1": np.mean(
                        [get_rotation_ratio(trained, f"block{i}_layer0", "1") for i in range(12)]
                    ),
                    "rotation_ratio_k2": np.mean(
                        [get_rotation_ratio(trained, f"block{i}_layer0", "2") for i in range(12)]
                    ),
                    "kl_k1": np.mean([get_kl(trained, f"block{i}_layer0", "1") for i in range(12)]),
                    "pr_k1": np.mean([get_pr(trained, f"block{i}_layer0", "1") for i in range(12)]),
                }

    return summary


def get_concentration_ratio(data: dict, block_key: str, k: str = "1", n_cols: int = 3072) -> float:
    """Get PR relative to uniform distribution.

    For uniform: PR = 1/sqrt(n)
    Concentration ratio = PR / PR_uniform = PR * sqrt(n)
    """
    pr = get_pr(data, block_key, k)
    pr_uniform = 1 / np.sqrt(n_cols)
    return pr / pr_uniform


# =============================================================================
# Timeline Analysis
# =============================================================================


def extract_timeline_series(
    run_data: dict,
    metric_fn,
    blocks: list[int] = None,
    layer: int = 0,
    k: str = "1",
    include_init: bool = True,
) -> tuple[list[int], dict[int, list[float]]]:
    """Extract a metric across all epochs for specified blocks.

    Returns:
        epochs: List of epoch numbers (0 = init if included)
        block_series: Dict mapping block_idx to list of metric values
    """
    if blocks is None:
        blocks = list(range(12))

    epochs = []
    block_series = {b: [] for b in blocks}

    # Init
    if include_init:
        init = run_data.get("full_init") or run_data.get("init")
        if init:
            epochs.append(0)
            for b in blocks:
                key = f"block{b}_layer{layer}"
                block_series[b].append(metric_fn(init, key, k))

    # Timeline
    if "timeline" in run_data:
        for epoch in sorted(run_data["timeline"].keys()):
            epochs.append(epoch)
            data = run_data["timeline"][epoch]
            for b in blocks:
                key = f"block{b}_layer{layer}"
                block_series[b].append(metric_fn(data, key, k))

    return epochs, block_series


def compute_change_from_init(
    run_data: dict,
    metric_fn,
    blocks: list[int] = None,
    layer: int = 0,
    k: str = "1",
) -> tuple[list[int], dict[int, list[float]]]:
    """Compute percentage change from init for each epoch.

    Returns:
        epochs: List of epoch numbers (excluding 0)
        block_changes: Dict mapping block_idx to list of % changes
    """
    epochs, block_series = extract_timeline_series(
        run_data, metric_fn, blocks, layer, k, include_init=True
    )

    if len(epochs) < 2:
        return [], {}

    block_changes = {}
    for b, values in block_series.items():
        init_val = values[0]
        if init_val != 0:
            block_changes[b] = [(v / init_val - 1) * 100 for v in values[1:]]
        else:
            block_changes[b] = [0.0] * (len(values) - 1)

    return epochs[1:], block_changes


# =============================================================================
# Visualization
# =============================================================================


def plot_timeline(
    run_data: dict,
    metric_fn,
    metric_name: str = "Metric",
    blocks: list[int] = None,
    layer: int = 0,
    k: str = "1",
    figsize: tuple[int, int] = (12, 6),
    show_change: bool = False,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot metric evolution across training.

    Args:
        run_data: Loaded run data dict
        metric_fn: Function to extract metric (e.g., get_rotation_ratio)
        metric_name: Name for plot title/labels
        blocks: Which blocks to plot (None = all 12)
        layer: MLP layer (0 or 1)
        k: Minor order ('1' or '2')
        figsize: Figure size
        show_change: If True, show % change from init; else show raw values
        save_path: Optional path to save figure
    """
    if blocks is None:
        blocks = list(range(12))

    fig, ax = plt.subplots(figsize=figsize)

    if show_change:
        epochs, block_data = compute_change_from_init(run_data, metric_fn, blocks, layer, k)
        ylabel = f"{metric_name} Change (%)"
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    else:
        epochs, block_data = extract_timeline_series(run_data, metric_fn, blocks, layer, k)
        ylabel = metric_name

    # Color map for blocks
    colors = plt.cm.viridis(np.linspace(0, 1, len(blocks)))

    for (b, values), color in zip(block_data.items(), colors):
        ax.plot(epochs, values, "o-", label=f"Block {b}", color=color, markersize=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric_name} (k={k}, layer{layer}) During Training")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_all_metrics_timeline(
    run_data: dict,
    blocks: list[int] = None,
    layer: int = 0,
    figsize: tuple[int, int] = (15, 12),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot all three KA metrics (rotation ratio, KL, PR) in one figure."""
    if blocks is None:
        blocks = [0, 3, 6, 9, 11]  # Representative blocks

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    metrics = [
        (get_rotation_ratio, "Rotation Ratio", "1"),
        (get_rotation_ratio, "Rotation Ratio", "2"),
        (get_kl, "KL Divergence", "1"),
        (get_kl, "KL Divergence", "2"),
        (get_pr, "Participation Ratio", "1"),
        (get_pr, "Participation Ratio", "2"),
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, len(blocks)))

    for idx, (metric_fn, name, k) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        epochs, block_data = compute_change_from_init(run_data, metric_fn, blocks, layer, k)

        for (b, values), color in zip(block_data.items(), colors):
            ax.plot(epochs, values, "o-", label=f"Block {b}", color=color, markersize=3)

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Change from Init (%)")
        ax.set_title(f"{name} k={k}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"KA Geometry Evolution (layer{layer})", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_layer_comparison(
    run_data: dict,
    metric_fn,
    metric_name: str = "Metric",
    blocks: list[int] = None,
    k: str = "1",
    epoch: int | None = None,
    figsize: tuple[int, int] = (10, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """Compare layer0 vs layer1 for a metric at a specific epoch."""
    if blocks is None:
        blocks = list(range(12))

    # Get data for the epoch
    if epoch is None:
        # Use latest epoch
        if "timeline" in run_data:
            epoch = max(run_data["timeline"].keys())
            data = run_data["timeline"][epoch]
        else:
            data = run_data.get("full_init") or run_data.get("init")
            epoch = 0
    elif epoch == 0:
        data = run_data.get("full_init") or run_data.get("init")
    else:
        data = run_data["timeline"][epoch]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(blocks))
    width = 0.35

    layer0_vals = [metric_fn(data, f"block{b}_layer0", k) for b in blocks]
    layer1_vals = [metric_fn(data, f"block{b}_layer1", k) for b in blocks]

    ax.bar(x - width / 2, layer0_vals, width, label="fc1 (layer0)")
    ax.bar(x + width / 2, layer1_vals, width, label="fc2 (layer1)")

    ax.set_xlabel("Block")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} (k={k}) at Epoch {epoch}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}" for b in blocks])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Report Generation
# =============================================================================


def generate_run_report(run_dir: str | Path, output_dir: str | None = None) -> str:
    """Generate a summary report for a run with figures.

    Args:
        run_dir: Path to run directory
        output_dir: Where to save figures (default: run_dir/report)

    Returns:
        Report text
    """
    run_dir = Path(run_dir)
    if output_dir is None:
        output_dir = run_dir / "report"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    run_data = load_run_data(run_dir)

    report = []
    report.append(f"# KA Geometry Report: {run_dir.name}")
    report.append("")

    # Summary stats
    summary = summarize_run(run_data)
    report.append("## Summary Statistics")
    report.append("")

    for key, stats in summary.items():
        report.append(f"### {key}")
        for metric, val in stats.items():
            report.append(f"  - {metric}: {val:.6f}")
        report.append("")

    # Generate plots
    if "timeline" in run_data:
        report.append("## Timeline Plots")
        report.append("")

        # All metrics plot
        fig = plot_all_metrics_timeline(
            run_data, save_path=str(output_dir / "all_metrics_timeline.png")
        )
        plt.close(fig)
        report.append("![All Metrics](all_metrics_timeline.png)")
        report.append("")

        # Individual metric plots
        for metric_fn, name in [
            (get_rotation_ratio, "Rotation Ratio"),
            (get_kl, "KL Divergence"),
            (get_pr, "Participation Ratio"),
        ]:
            fig = plot_timeline(
                run_data,
                metric_fn,
                name,
                show_change=True,
                save_path=str(output_dir / f"{name.lower().replace(' ', '_')}_timeline.png"),
            )
            plt.close(fig)

    return "\n".join(report)


# =============================================================================
# Quick Analysis Functions
# =============================================================================


def quick_compare(
    run_dir: str | Path,
    epoch: int = None,
    blocks: list[int] = None,
) -> None:
    """Quick comparison of init vs trained for all metrics."""
    run_data = load_run_data(run_dir)

    if "timeline" not in run_data:
        print("No timeline data found")
        return

    if epoch is None:
        epoch = max(run_data["timeline"].keys())

    init = run_data.get("full_init") or run_data.get("init")
    trained = run_data["timeline"][epoch]

    if blocks is None:
        blocks = list(range(12))

    print(f"\n{'=' * 70}")
    print(f"Comparison: Init vs Epoch {epoch}")
    print(f"{'=' * 70}\n")

    for k in ["1", "2"]:
        print(f"\n--- k={k} ---\n")

        for metric_fn, name in [
            (get_rotation_ratio, "Rotation Ratio"),
            (get_kl, "KL Divergence"),
            (get_pr, "PR (L2/L1)"),
        ]:
            results = compare_metrics(init, trained, blocks, k=k, metric_fn=metric_fn)
            print(f"{name}:")
            print(f"  Init avg: {results['avg_init']:.6f}")
            print(f"  Trained avg: {results['avg_trained']:.6f}")
            print(f"  Change: {results['avg_change_pct']:+.1f}%")
            print()


if __name__ == "__main__":
    # Example usage
    runs = list_runs()
    if runs:
        print(f"Found {len(runs)} runs:")
        for r in runs[-5:]:
            print(f"  - {r.name}")

        # Quick analysis of latest run
        latest = runs[-1]
        print(f"\nAnalyzing: {latest.name}")
        quick_compare(latest)
