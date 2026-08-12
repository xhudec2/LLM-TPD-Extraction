#!/usr/bin/env python3
"""Standalone Convergence Curve Visualization Tool"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def scan_rounds(input_dir: Path) -> List[int]:
    """Find all round_N directories and return sorted list of round numbers"""
    rounds = []
    for item in input_dir.iterdir():
        if item.is_dir() and item.name.startswith("round_"):
            try:
                round_num = int(item.name.split("_")[1])
                rounds.append(round_num)
            except (IndexError, ValueError):
                print(
                    f"Warning: Could not parse round number from directory: {item.name}"
                )

    return sorted(rounds)


def unescape_doi(paper_dirname: str) -> str:
    """Convert escaped DOI directory name to actual DOI"""
    if paper_dirname.startswith("paper_"):
        doi = paper_dirname[6:]
    else:
        doi = paper_dirname

    parts = doi.split("_", 2)
    if len(parts) >= 2:
        doi = f"{parts[0]}/{parts[1]}"
        if len(parts) == 3:
            doi = f"{doi}.{parts[2]}"

    return doi


def load_per_paper_metrics(
    input_dir: Path, rounds: List[int]
) -> Dict[str, Dict[int, Dict]]:
    """Load per-paper metrics from record_based_evaluation_results"""
    per_paper_data = {}

    for round_num in rounds:
        round_dir = input_dir / f"round_{round_num}"
        batch_results_dir = round_dir / "batch_results"

        if not batch_results_dir.exists():
            print(f"Warning: batch_results not found in round {round_num}")
            continue

        for paper_dir in batch_results_dir.iterdir():
            if not paper_dir.is_dir() or not paper_dir.name.startswith("paper_"):
                continue

            doi = unescape_doi(paper_dir.name)

            eval_file = paper_dir / "record_based_evaluation_results.json"
            if not eval_file.exists():
                print(f"Warning: Evaluation file not found: {eval_file}")
                continue

            try:
                with open(eval_file, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)

                part1 = eval_data.get("part1_record_level", {})
                part2 = eval_data.get("part2_entry_level", {})

                metrics = {
                    "part1_precision": part1.get("precision_records", 0.0),
                    "part1_recall": part1.get("recall_records", 0.0),
                    "part2_precision": part2.get("precision_entry", 0.0),
                    "part2_recall": part2.get("recall_entry", 0.0),
                }

                if doi not in per_paper_data:
                    per_paper_data[doi] = {}

                per_paper_data[doi][round_num] = metrics

            except Exception as e:
                print(f"Error loading {eval_file}: {e}")
                continue

    return per_paper_data


def load_aggregate_metrics(input_dir: Path, rounds: List[int]) -> Dict[int, Dict]:
    """Load aggregate metrics from aggregate_metrics"""
    aggregate_data = {}

    for round_num in rounds:
        round_dir = input_dir / f"round_{round_num}"
        agg_file = round_dir / "aggregate_metrics.json"

        if not agg_file.exists():
            print(f"Warning: aggregate_metrics.json not found for round {round_num}")
            continue

        try:
            with open(agg_file, "r", encoding="utf-8") as f:
                agg_data = json.load(f)

            aggregate_data[round_num] = {
                "avg_part1_precision": agg_data.get("avg_part1_precision", 0.0),
                "avg_part1_recall": agg_data.get("avg_part1_recall", 0.0),
                "avg_part2_precision": agg_data.get("avg_part2_precision", 0.0),
                "avg_part2_recall": agg_data.get("avg_part2_recall", 0.0),
                "num_papers": agg_data.get("num_papers", 0),
            }

        except Exception as e:
            print(f"Error loading {agg_file}: {e}")
            continue

    return aggregate_data


def compute_aggregate_from_papers(
    per_paper_data: Dict[str, Dict[int, Dict]], round_num: int
) -> Dict:
    """Compute batch-level aggregate metrics from individual paper metrics"""
    part1_precisions = []
    part1_recalls = []
    part2_precisions = []
    part2_recalls = []

    for doi, rounds_data in per_paper_data.items():
        if round_num in rounds_data:
            metrics = rounds_data[round_num]
            part1_precisions.append(metrics["part1_precision"])
            part1_recalls.append(metrics["part1_recall"])
            part2_precisions.append(metrics["part2_precision"])
            part2_recalls.append(metrics["part2_recall"])

    if not part1_precisions:
        return {
            "avg_part1_precision": 0.0,
            "avg_part1_recall": 0.0,
            "avg_part2_precision": 0.0,
            "avg_part2_recall": 0.0,
            "std_part1_precision": 0.0,
            "std_part1_recall": 0.0,
            "std_part2_precision": 0.0,
            "std_part2_recall": 0.0,
            "num_papers": 0,
        }

    return {
        "avg_part1_precision": np.mean(part1_precisions),
        "avg_part1_recall": np.mean(part1_recalls),
        "avg_part2_precision": np.mean(part2_precisions),
        "avg_part2_recall": np.mean(part2_recalls),
        "std_part1_precision": np.std(part1_precisions),
        "std_part1_recall": np.std(part1_recalls),
        "std_part2_precision": np.std(part2_precisions),
        "std_part2_recall": np.std(part2_recalls),
        "num_papers": len(part1_precisions),
    }


def parse_rounds_filter(rounds_str: str, available_rounds: List[int]) -> List[int]:
    """Parse rounds filter string and return filtered list"""
    if "-" in rounds_str and "," not in rounds_str:
        try:
            start, end = map(int, rounds_str.split("-"))
            filtered = [r for r in available_rounds if start <= r <= end]
        except ValueError:
            print(f"Warning: Invalid range format: {rounds_str}")
            return available_rounds
    else:
        try:
            requested = list(map(int, rounds_str.split(",")))
            filtered = [r for r in available_rounds if r in requested]
        except ValueError:
            print(f"Warning: Invalid rounds format: {rounds_str}")
            return available_rounds

    return sorted(filtered)


def filter_papers_by_pattern(per_paper_data: Dict, pattern: str) -> Dict:
    """Filter papers by regex pattern on DOI"""
    try:
        regex = re.compile(pattern)
        filtered = {
            doi: data for doi, data in per_paper_data.items() if regex.search(doi)
        }
        return filtered
    except re.error as e:
        print(f"Warning: Invalid regex pattern '{pattern}': {e}")
        return per_paper_data


def validate_aggregate_consistency(
    aggregate_data: Dict[int, Dict], per_paper_data: Dict[str, Dict[int, Dict]]
):
    """Validate consistency between aggregate metrics and computed values from..."""
    print("\nValidating aggregate consistency...")

    for round_num in aggregate_data:
        agg_metrics = aggregate_data[round_num]
        computed_metrics = compute_aggregate_from_papers(per_paper_data, round_num)

        metrics_to_check = [
            "avg_part1_precision",
            "avg_part1_recall",
            "avg_part2_precision",
            "avg_part2_recall",
        ]

        for metric in metrics_to_check:
            agg_value = agg_metrics.get(metric, 0.0)
            computed_value = computed_metrics.get(metric, 0.0)
            diff = abs(agg_value - computed_value)

            if diff > 0.001:
                print(
                    f"  Warning: Round {round_num}, {metric}: "
                    f"aggregate={agg_value:.4f}, computed={computed_value:.4f}, diff={diff:.4f}"
                )

    print("  Validation complete")


def configure_matplotlib():
    """Configure matplotlib for consistent plotting"""
    plt.rcParams["figure.figsize"] = (14, 10)
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 8


def plot_per_paper_curves(
    per_paper_data: Dict[str, Dict[int, Dict]],
    output_dir: Path,
    threshold: float = 0.9,
    dpi: int = 150,
    figsize: Tuple[float, float] = (14, 10),
) -> Tuple[Path, Path]:
    """Generate per-paper convergence curves (2×2 subplot grid)"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Per-Paper Convergence Curves", fontsize=16, fontweight="bold")

    metrics_specs = [
        ("part1_precision", "Part 1 (Record-level) Precision"),
        ("part1_recall", "Part 1 (Record-level) Recall"),
        ("part2_precision", "Part 2 (Field-level) Precision"),
        ("part2_recall", "Part 2 (Field-level) Recall"),
    ]

    all_rounds = set()
    for rounds_data in per_paper_data.values():
        all_rounds.update(rounds_data.keys())
    all_rounds = sorted(all_rounds)

    colors = plt.cm.tab10(np.linspace(0, 1, len(per_paper_data)))

    for idx, (metric_key, metric_title) in enumerate(metrics_specs):
        ax = axes[idx // 2, idx % 2]

        for color_idx, (doi, rounds_data) in enumerate(per_paper_data.items()):
            rounds = sorted(rounds_data.keys())
            values = [rounds_data[r].get(metric_key, 0.0) for r in rounds]

            short_doi = doi[-30:] if len(doi) > 30 else doi

            ax.plot(
                rounds,
                values,
                marker="o",
                color=colors[color_idx],
                alpha=0.7,
                linewidth=2,
                markersize=6,
                label=short_doi,
            )

        if all_rounds:
            ax.axhline(
                y=threshold,
                color="blue",
                linestyle="--",
                linewidth=2,
                label=f"Threshold ({threshold})",
            )

        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel("Metric Value", fontsize=11)
        ax.set_title(metric_title, fontsize=12, fontweight="bold")
        ax.set_ylim([0.4, 1.2])

        if all_rounds:
            ax.set_xticks(all_rounds)
            ax.set_xticklabels([str(r) for r in all_rounds])

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)

    plt.tight_layout()

    png_path = output_dir / "per_paper_convergence.png"
    pdf_path = output_dir / "per_paper_convergence.pdf"

    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    return png_path, pdf_path


def plot_aggregate_curves(
    aggregate_data: Dict[int, Dict],
    output_dir: Path,
    threshold: float = 0.9,
    show_std: bool = True,
    dpi: int = 150,
    figsize: Tuple[float, float] = (14, 10),
) -> Tuple[Path, Path]:
    """Generate aggregate convergence curves (2×2 subplot grid)"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Aggregate Convergence Curves", fontsize=16, fontweight="bold")

    metrics_specs = [
        (
            "avg_part1_precision",
            "std_part1_precision",
            "Part 1 (Record-level) Precision",
        ),
        ("avg_part1_recall", "std_part1_recall", "Part 1 (Record-level) Recall"),
        (
            "avg_part2_precision",
            "std_part2_precision",
            "Part 2 (Field-level) Precision",
        ),
        ("avg_part2_recall", "std_part2_recall", "Part 2 (Field-level) Recall"),
    ]

    rounds = sorted(aggregate_data.keys())

    for idx, (metric_key, std_key, metric_title) in enumerate(metrics_specs):
        ax = axes[idx // 2, idx % 2]

        values = [aggregate_data[r].get(metric_key, 0.0) for r in rounds]

        ax.plot(
            rounds,
            values,
            marker="o",
            color="darkgreen",
            linewidth=3,
            markersize=8,
            label="Average",
            zorder=3,
        )

        if show_std:
            stds = [aggregate_data[r].get(std_key, 0.0) for r in rounds]
            if any(s > 0 for s in stds):
                upper = [v + s for v, s in zip(values, stds)]
                lower = [max(0, v - s) for v, s in zip(values, stds)]
                ax.fill_between(
                    rounds,
                    lower,
                    upper,
                    color="darkgreen",
                    alpha=0.2,
                    label="±1 std dev",
                )

        if rounds:
            ax.axhline(
                y=threshold,
                color="blue",
                linestyle="--",
                linewidth=2,
                label=f"Threshold ({threshold})",
            )

        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel("Metric Value", fontsize=11)
        ax.set_title(metric_title, fontsize=12, fontweight="bold")
        ax.set_ylim([0.4, 1.2])

        if rounds:
            ax.set_xticks(rounds)
            ax.set_xticklabels([str(r) for r in rounds])

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)

    plt.tight_layout()

    png_path = output_dir / "aggregate_convergence.png"
    pdf_path = output_dir / "aggregate_convergence.pdf"

    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    return png_path, pdf_path


def compute_summary_statistics(
    per_paper_data: Dict[str, Dict[int, Dict]],
    aggregate_data: Dict[int, Dict],
    rounds: List[int],
) -> Dict:
    """Compute summary statistics for reporting"""
    num_rounds = len(rounds)
    num_papers = len(per_paper_data)
    paper_list = sorted(per_paper_data.keys())

    final_round = max(rounds)
    final_metrics = aggregate_data.get(final_round, {})

    first_round = min(rounds)
    first_metrics = aggregate_data.get(first_round, {})

    improvements = {}
    for metric in [
        "avg_part1_precision",
        "avg_part1_recall",
        "avg_part2_precision",
        "avg_part2_recall",
    ]:
        final_val = final_metrics.get(metric, 0.0)
        first_val = first_metrics.get(metric, 0.0)
        improvements[metric] = final_val - first_val

    return {
        "num_rounds": num_rounds,
        "rounds_range": (min(rounds), max(rounds)),
        "num_papers": num_papers,
        "paper_list": paper_list,
        "final_metrics": final_metrics,
        "first_metrics": first_metrics,
        "improvements": improvements,
    }


def print_summary(stats: Dict, output_dir: Path):
    """Print formatted summary to console"""
    print("\n" + "=" * 80)
    print("Convergence Visualization Summary")
    print("=" * 80)
    print(f"Output Directory: {output_dir}")
    print()

    print("Data Overview:")
    rounds_range = stats["rounds_range"]
    print(
        f"  - Rounds processed: {rounds_range[0]}-{rounds_range[1]} ({stats['num_rounds']} total)"
    )
    print(f"  - Papers processed: {stats['num_papers']}")

    paper_list = stats["paper_list"]
    if len(paper_list) <= 5:
        print(f"  - Papers: {', '.join(paper_list)}")
    else:
        print(
            f"  - Papers: {', '.join(paper_list[:3])}, ... (+{len(paper_list) - 3} more)"
        )
    print()

    final_metrics = stats["final_metrics"]
    first_metrics = stats["first_metrics"]
    improvements = stats["improvements"]

    print(f"Final Metrics (Round {rounds_range[1]}):")

    def format_change(value):
        if value > 0:
            return f"↑ +{value:.3f}"
        elif value < 0:
            return f"↓ {value:.3f}"
        else:
            return "→ +0.000"

    metrics_display = [
        ("avg_part1_precision", "Part 1 Precision"),
        ("avg_part1_recall", "Part 1 Recall"),
        ("avg_part2_precision", "Part 2 Precision"),
        ("avg_part2_recall", "Part 2 Recall"),
    ]

    for metric_key, metric_name in metrics_display:
        final_val = final_metrics.get(metric_key, 0.0)
        change = improvements.get(metric_key, 0.0)
        print(
            f"  {metric_name}: {final_val:.3f}  ({format_change(change)} from Round {rounds_range[0]})"
        )
    print()

    print("Output Files Generated:")
    print("  - per_paper_convergence.png")
    print("  - per_paper_convergence.pdf (vector)")
    print("  - aggregate_convergence.png")
    print("  - aggregate_convergence.pdf (vector)")
    print("  - visualization_metadata.json")
    print("=" * 80 + "\n")


def save_metadata(
    stats: Dict,
    input_dir: Path,
    output_dir: Path,
    threshold: float,
    rounds_filter: Optional[str],
    papers_filter: Optional[str],
):
    """Save visualization metadata to JSON file"""
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "rounds_processed": list(
            range(stats["rounds_range"][0], stats["rounds_range"][1] + 1)
        ),
        "papers_processed": stats["num_papers"],
        "paper_dois": stats["paper_list"],
        "threshold": threshold,
        "filters_applied": {
            "rounds": rounds_filter,
            "papers": papers_filter,
        },
        "final_metrics": stats["final_metrics"],
        "script_version": "1.0.0",
    }

    metadata_path = output_dir / "visualization_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate convergence curve visualizations for Active Prompting batch results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/capo_visualize.py \\
    --input-dir output/results/active_prompting_batch_251127_2008

  # Custom output and threshold
  python scripts/capo_visualize.py \\
    --input-dir output/results/active_prompting_batch_251127_2008 \\
    --output-dir ~/Desktop/convergence_viz/ \\
    --threshold 0.95

  # Only rounds 1-3
  python scripts/capo_visualize.py \\
    --input-dir output/results/active_prompting_batch_251127_2008 \\
    --rounds 1-3

  # Only aggregate plot (skip per-paper)
  python scripts/capo_visualize.py \\
    --input-dir output/results/active_prompting_batch_251127_2008 \\
    --skip-per-paper

  # High resolution for publication
  python scripts/capo_visualize.py \\
    --input-dir output/results/active_prompting_batch_251127_2008 \\
    --dpi 300
""",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to active_prompting_batch_* directory",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: {input_dir}/visualizations/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Threshold line value (default: 0.9)",
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="PNG resolution (default: 150)"
    )

    parser.add_argument(
        "--rounds",
        type=str,
        default=None,
        help='Comma-separated rounds or range (e.g., "1,2,3" or "1-5")',
    )
    parser.add_argument(
        "--papers", type=str, default=None, help="Regex pattern to filter papers by DOI"
    )

    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Disable standard deviation shading in aggregate plot",
    )
    parser.add_argument(
        "--skip-per-paper", action="store_true", help="Skip per-paper visualization"
    )
    parser.add_argument(
        "--skip-aggregate", action="store_true", help="Skip aggregate visualization"
    )

    parser.add_argument(
        "--use-aggregate",
        action="store_true",
        help="Prefer aggregate_metrics.json over computing from papers",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate consistency between aggregate and per-paper metrics",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir = (
        Path(args.output_dir) if args.output_dir else input_dir / "visualizations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nConvergence Visualization Tool")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}\n")

    configure_matplotlib()

    print("Scanning for rounds...")
    rounds = scan_rounds(input_dir)
    if not rounds:
        print(f"Error: No round directories found in {input_dir}")
        sys.exit(1)

    if args.rounds:
        rounds = parse_rounds_filter(args.rounds, rounds)
        if not rounds:
            print(f"Error: No rounds match the filter: {args.rounds}")
            sys.exit(1)

    print(f"Found rounds: {rounds}\n")

    print("Loading per-paper metrics...")
    per_paper_data = load_per_paper_metrics(input_dir, rounds)

    if not per_paper_data:
        print("Error: No paper data found")
        sys.exit(1)

    if args.papers:
        per_paper_data = filter_papers_by_pattern(per_paper_data, args.papers)
        if not per_paper_data:
            print(f"Error: No papers match the filter: {args.papers}")
            sys.exit(1)

    print(f"Loaded data for {len(per_paper_data)} papers\n")

    if args.use_aggregate:
        print("Loading aggregate metrics from JSON...")
        aggregate_data = load_aggregate_metrics(input_dir, rounds)

        for round_num in rounds:
            if round_num not in aggregate_data:
                print(
                    f"  Computing aggregate for round {round_num} (not found in JSON)"
                )
                aggregate_data[round_num] = compute_aggregate_from_papers(
                    per_paper_data, round_num
                )
    else:
        print("Computing aggregate metrics from per-paper data...")
        aggregate_data = {}
        for round_num in rounds:
            aggregate_data[round_num] = compute_aggregate_from_papers(
                per_paper_data, round_num
            )

    if args.validate:
        validate_aggregate_consistency(aggregate_data, per_paper_data)

    if not args.skip_per_paper:
        print("\nGenerating per-paper convergence curves...")
        png_path, pdf_path = plot_per_paper_curves(
            per_paper_data, output_dir, args.threshold, args.dpi
        )
        print(f"  ✓ Saved: {png_path.name}")
        print(f"  ✓ Saved: {pdf_path.name}")

    if not args.skip_aggregate:
        print("\nGenerating aggregate convergence curves...")
        png_path, pdf_path = plot_aggregate_curves(
            aggregate_data,
            output_dir,
            args.threshold,
            show_std=not args.no_std,
            dpi=args.dpi,
        )
        print(f"  ✓ Saved: {png_path.name}")
        print(f"  ✓ Saved: {pdf_path.name}")

    stats = compute_summary_statistics(per_paper_data, aggregate_data, rounds)
    print_summary(stats, output_dir)

    save_metadata(
        stats, input_dir, output_dir, args.threshold, args.rounds, args.papers
    )

    print("✓ Visualization complete!\n")


if __name__ == "__main__":
    main()
