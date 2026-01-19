#!/usr/bin/env python3
"""
Plot evaluation results from lighteval output directories.

Usage:
    python plot.py /path/to/results/model1 /path/to/results/model2 ...
    python plot.py /path/to/results/model1 /path/to/results/model2 --legend_names "Model 1" "Model 2"
    python plot.py /path/to/results/model1 --output /path/to/output.png --title "My Results"
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


# Default color palette matching FineWeb2 paper style
DEFAULT_COLORS = [
    '#c41e3a',  # deep rose/magenta
    '#008080',  # teal
    '#5bc0be',  # light cyan
    '#7f7f7f',  # gray
    '#e07b39',  # orange
    '#6b5b95',  # purple
    '#88b04b',  # green
    '#f7cac9',  # pink
]

# Default tokens per step based on typical config:
# dp=8, micro_batch_size=4, batch_accumulation_per_replica=32, sequence_length=2048
# = 8 * 4 * 32 * 2048 = 2,097,152
DEFAULT_TOKENS_PER_STEP = 2_097_152


def setup_plot_style():
    """Set up matplotlib style to match FineWeb2 paper."""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.linewidth'] = 0.7


def extract_step_number(step_dir: str) -> Optional[int]:
    """Extract step number from directory name like 'step_1200'."""
    match = re.search(r'step_(\d+)', step_dir)
    if match:
        return int(match.group(1))
    return None


def find_result_files(step_path: Path) -> list[Path]:
    """Recursively find all JSON result files in a step directory."""
    result_files = []

    # Walk through all subdirectories
    for root, dirs, files in os.walk(step_path):
        root_path = Path(root)
        for f in files:
            if f.endswith('.json'):
                result_files.append(root_path / f)

    return result_files


def load_results_from_step(step_path: Path) -> Optional[float]:
    """Load and aggregate results from a step directory.

    Looks for lighteval result JSON files and computes aggregate score.
    Searches all subdirectories recursively.
    """
    results_files = find_result_files(step_path)

    if not results_files:
        return None

    all_scores = []

    for results_file in results_files:
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)

            # Handle different lighteval output formats
            if 'results' in data:
                results = data['results']
                for task_name, task_results in results.items():
                    # Look for accuracy or normalized accuracy metrics
                    for metric in ['acc_norm', 'acc', 'exact_match', 'f1', 'quasi_exact_match', 'quasi_f1']:
                        if metric in task_results:
                            score = task_results[metric]
                            if isinstance(score, (int, float)) and 0 <= score <= 1:
                                all_scores.append(score)
                            break
            elif isinstance(data, dict):
                # Try to find scores in flat structure
                for key, value in data.items():
                    if isinstance(value, (int, float)) and 0 <= value <= 1:
                        all_scores.append(value)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Could not parse {results_file}: {e}")
            continue

    if all_scores:
        return np.mean(all_scores)
    return None


def load_model_results(model_dir: Path, tokens_per_step: float = DEFAULT_TOKENS_PER_STEP) -> tuple[list[float], list[float]]:
    """Load all results for a model directory.

    Args:
        model_dir: Path to model results directory containing step_* subdirs
        tokens_per_step: Number of tokens per training step

    Returns:
        Tuple of (tokens_list, scores_list) sorted by tokens
    """
    tokens = []
    scores = []

    # Find all step directories
    step_dirs = sorted(model_dir.glob('step_*'))

    if not step_dirs:
        print(f"  No step_* directories found in {model_dir}")
        return [], []

    print(f"  Found {len(step_dirs)} step directories in {model_dir.name}")

    for step_dir in step_dirs:
        step_num = extract_step_number(step_dir.name)
        if step_num is None:
            continue

        score = load_results_from_step(step_dir)
        if score is not None:
            # Convert step to tokens (in billions)
            tokens_billions = (step_num * tokens_per_step) / 1e9
            tokens.append(tokens_billions)
            scores.append(score)
            print(f"    Step {step_num} ({tokens_billions:.1f}B tokens): score = {score:.4f}")
        else:
            print(f"    Step {step_num}: no results found")

    # Sort by tokens
    if tokens:
        sorted_pairs = sorted(zip(tokens, scores))
        tokens, scores = zip(*sorted_pairs)
        return list(tokens), list(scores)

    return [], []


def plot_results(
    model_dirs: list[Path],
    legend_names: Optional[list[str]] = None,
    output_path: str = 'evaluation_results.png',
    title: str = 'Evaluation Results',
    tokens_per_step: float = DEFAULT_TOKENS_PER_STEP,
    colors: Optional[list[str]] = None,
):
    """Plot evaluation results from multiple model directories.

    Args:
        model_dirs: List of paths to model result directories
        legend_names: Optional custom legend names (same order as model_dirs)
        output_path: Output file path (supports .png and .pdf)
        title: Chart title
        tokens_per_step: Number of tokens per training step
        colors: Optional custom colors for each model
    """
    setup_plot_style()

    if colors is None:
        colors = DEFAULT_COLORS

    if legend_names is None:
        legend_names = [d.name for d in model_dirs]

    print(f"Tokens per step: {tokens_per_step:,} ({tokens_per_step/1e9:.4f}B)")
    print()

    # Load all model results
    model_data = {}
    for model_dir, name in zip(model_dirs, legend_names):
        print(f"Loading results for {name}:")
        tokens, scores = load_model_results(model_dir, tokens_per_step)
        if tokens and scores:
            model_data[name] = (tokens, scores)
        else:
            print(f"  Warning: No valid results found")
        print()

    if not model_data:
        print("Error: No valid results found in any directory")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each model
    for i, (name, (tokens, scores)) in enumerate(model_data.items()):
        color = colors[i % len(colors)]
        ax.plot(tokens, scores, color=color, linewidth=3.2, solid_capstyle='round', label=name)
        ax.plot(tokens[-1], scores[-1], 'o', color=color, markersize=8)

    # Styling to match FineWeb2
    ax.set_xlabel('Tokens seen', fontsize=12)
    ax.set_ylabel('Aggregate Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='normal')

    # Grid styling - dotted light gray lines
    ax.grid(True, linestyle=':', alpha=0.6, color='#aaaaaa')
    ax.set_axisbelow(True)

    # Auto-scale axes with padding
    all_tokens = [t for tokens, _ in model_data.values() for t in tokens]
    all_scores = [s for _, scores in model_data.values() for s in scores]

    if all_tokens and all_scores:
        x_min, x_max = min(all_tokens), max(all_tokens)
        y_min, y_max = min(all_scores), max(all_scores)

        x_padding = (x_max - x_min) * 0.1 or 1
        y_padding = (y_max - y_min) * 0.1 or 0.01

        ax.set_xlim(max(0, x_min - x_padding), x_max + x_padding)
        ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)

    # Format x-axis ticks as "XB" for billions
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}B'))

    # Legend with visible border
    ax.legend(loc='upper left', frameon=True, fancybox=False,
              edgecolor='#000000', fontsize=10, framealpha=1.0)

    # Box border styling
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#000000')
        spine.set_linewidth(1.2)

    # Tick styling
    ax.tick_params(colors='#333333', direction='out')

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor='white', edgecolor='none', bbox_inches='tight')

    # Also save PDF if PNG was requested
    if output_path.suffix == '.png':
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, facecolor='white', edgecolor='none', bbox_inches='tight')
        print(f"Saved: {output_path} and {pdf_path}")
    else:
        print(f"Saved: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot evaluation results from lighteval output directories.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'directories',
        nargs='+',
        type=Path,
        help='Result directories to plot (e.g., /scratch/nanotron_results/results/model1)'
    )
    parser.add_argument(
        '--legend_names',
        nargs='+',
        help='Custom legend names for each directory (same order as directories)'
    )
    parser.add_argument(
        '--output', '-o',
        default='evaluation_results.png',
        help='Output file path (default: evaluation_results.png)'
    )
    parser.add_argument(
        '--title', '-t',
        default='Evaluation Results',
        help='Chart title (default: Evaluation Results)'
    )
    parser.add_argument(
        '--tokens_per_step',
        type=float,
        default=DEFAULT_TOKENS_PER_STEP,
        help=f'Number of tokens per training step (default: {DEFAULT_TOKENS_PER_STEP:,})'
    )
    parser.add_argument(
        '--colors',
        nargs='+',
        help='Custom colors for each model (hex codes)'
    )

    args = parser.parse_args()

    # Validate legend_names count matches directories
    if args.legend_names and len(args.legend_names) != len(args.directories):
        parser.error(f'Number of legend_names ({len(args.legend_names)}) must match '
                     f'number of directories ({len(args.directories)})')

    # Validate colors count if provided
    if args.colors and len(args.colors) != len(args.directories):
        parser.error(f'Number of colors ({len(args.colors)}) must match '
                     f'number of directories ({len(args.directories)})')

    # Validate directories exist
    for d in args.directories:
        if not d.exists():
            parser.error(f'Directory does not exist: {d}')

    plot_results(
        model_dirs=args.directories,
        legend_names=args.legend_names,
        output_path=args.output,
        title=args.title,
        tokens_per_step=args.tokens_per_step,
        colors=args.colors,
    )


if __name__ == '__main__':
    main()
