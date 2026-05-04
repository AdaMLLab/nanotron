#!/usr/bin/env python3
"""
Generate FineTasks Arabic evaluation plots for aratrain continuation experiments.

Uses FineWeb2 methodology:
- Rescaling: (score - random_baseline) / (1 - random_baseline)
- Category grouping: GK, RC, RES, NLU
- Two-step aggregation: avg within category, then avg across categories
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Add the archive plotting code's parent to path for potential reuse
ARCHIVE_PLOTTING = Path("/home/alrashsm/github/nanotron-260214-archive/kaust/aramix_paper/evaluations/plotting")

# =============================================================================
# Configuration
# =============================================================================

TOKENS_PER_STEP = 8 * 4 * 32 * 2048  # = 2,097,152
OUTPUT_DIR = Path(os.environ.get("ARATRAIN_PLOT_OUTPUT", str(Path(__file__).parent)))
RESULTS_BASE = Path(os.path.expanduser("~/nanotron_checkpoint_results/aratrain/results"))
H100_RESULTS = Path(os.path.expanduser("~/nanotron_checkpoint_results_from_h100s/results"))

# =============================================================================
# Model Definitions
# =============================================================================

# Baseline (100% AraMix-HQ 30B from aramix paper)
BASELINE_MODEL = {
    "__baseline_aramix_hq__": {"display_name": "100% AraMix-HQ (baseline)", "path": H100_RESULTS / "aramix_hq"},
}

# Pure continuation runs (100% single dataset)
PURE_MODELS = {
    "finewiki_ar_checked_25bt": {"display_name": "FineWiki-AR"},
    "ultradata_math_25bt": {"display_name": "Math 50/50"},
    "openresearcher_25bt": {"display_name": "OpenResearcher"},
    "ultradata_math_qa_only_25bt": {"display_name": "Math QA"},
    "ultradata_math_textbook_only_25bt": {"display_name": "Math Textbook"},
}

# Blended runs (aramix-hq + other)
BLEND_MODELS = {
    "aramix_math_blend_25bt": {"display_name": "50% AraMix + 25% Txtbk + 25% QA"},
    "aramix_nyu_aco_ocr_25bt": {"display_name": "85% AraMix + 15% OCR"},
}

# 50/50 blend runs (excluding 50% OCR per user request)
BLEND_50_50_MODELS = {
    "aramix50_openresearcher50_25bt": {"display_name": "50% AraMix + 50% OpenResearcher"},
    "aramix50_finewiki50_25bt": {"display_name": "50% AraMix + 50% FineWiki"},
    "aramix50_math_qa50_25bt": {"display_name": "50% AraMix + 50% Math QA"},
    "aramix50_math_textbook50_25bt": {"display_name": "50% AraMix + 50% Math Textbook"},
}

# IBEX experiments
IBEX_DCLM_MODELS = {
    "dclm_pro_ibex_100_25bt": {"display_name": "100% DCLM-Pro-IBEX"},
    "aramix50_dclm_pro_ibex50_25bt": {"display_name": "50% AraMix + 50% DCLM-Pro"},
    "aramix25_dclm_pro_ibex75_25bt": {"display_name": "25% AraMix + 75% DCLM-Pro"},
    "aramix75_dclm_pro_ibex25_25bt": {"display_name": "75% AraMix + 25% DCLM-Pro"},
}

IBEX_FWE_MODELS = {
    "fineweb_edu_ibex_100_25bt": {"display_name": "100% FineWeb-Edu-IBEX"},
    "aramix50_fineweb_edu_ibex50_25bt": {"display_name": "50% AraMix + 50% FWE-IBEX"},
    "aramix25_fineweb_edu_ibex75_25bt": {"display_name": "25% AraMix + 75% FWE-IBEX"},
    "aramix75_fineweb_edu_ibex25_25bt": {"display_name": "75% AraMix + 25% FWE-IBEX"},
}

IBEX_ALL_MODELS = {**IBEX_DCLM_MODELS, **IBEX_FWE_MODELS}

# All models combined
ALL_MODELS = {**BASELINE_MODEL, **PURE_MODELS, **BLEND_MODELS, **BLEND_50_50_MODELS, **IBEX_ALL_MODELS}

# Plot groupings
PLOT_CONFIGS = {
    "all_models": {
        "models": ALL_MODELS,
        "title": "AraTrain: All Continuation Experiments",
        "basename": "finetasks_arabic_all",
    },
    "pure_vs_blends": {
        "models": {
            **BASELINE_MODEL,
            **{k: v for k, v in PURE_MODELS.items() if k in ["finewiki_ar_checked_25bt", "openresearcher_25bt", "ultradata_math_25bt"]},
            **BLEND_MODELS,
        },
        "title": "AraTrain: Pure Datasets vs Blends",
        "basename": "finetasks_arabic_pure_vs_blends",
    },
    "50_50_blends": {
        "models": {**BASELINE_MODEL, **BLEND_50_50_MODELS},
        "title": "AraTrain: 50/50 AraMix-HQ Blends",
        "basename": "finetasks_arabic_50_50_blends",
    },
    "ocr_comparison": {
        "models": {
            **BASELINE_MODEL,
            "aramix_nyu_aco_ocr_25bt": {"display_name": "85% AraMix + 15% OCR"},
        },
        "title": "AraTrain: OCR Blend Comparison",
        "basename": "finetasks_arabic_ocr_comparison",
    },
    "math_comparison": {
        "models": {
            **BASELINE_MODEL,
            "ultradata_math_qa_only_25bt": {"display_name": "100% Math QA"},
            "ultradata_math_textbook_only_25bt": {"display_name": "100% Math Textbook"},
            "ultradata_math_25bt": {"display_name": "50% QA + 50% Textbook"},
            "aramix_math_blend_25bt": {"display_name": "50% AraMix + 25% Txtbk + 25% QA"},
            "aramix50_math_qa50_25bt": {"display_name": "50% AraMix + 50% Math QA"},
            "aramix50_math_textbook50_25bt": {"display_name": "50% AraMix + 50% Math Textbook"},
        },
        "title": "AraTrain: Math Dataset Comparison",
        "basename": "finetasks_arabic_math_comparison",
    },
    "ibex_all": {
        "models": {**BASELINE_MODEL, **IBEX_ALL_MODELS},
        "title": "AraTrain: IBEX Experiments (DCLM-Pro + FineWeb-Edu)",
        "basename": "finetasks_arabic_ibex_all",
    },
    "ibex_dclm": {
        "models": {**BASELINE_MODEL, **IBEX_DCLM_MODELS},
        "title": "AraTrain: DCLM-Pro-IBEX Mix Comparison",
        "basename": "finetasks_arabic_ibex_dclm",
    },
    "ibex_fwe": {
        "models": {**BASELINE_MODEL, **IBEX_FWE_MODELS},
        "title": "AraTrain: FineWeb-Edu-IBEX Mix Comparison",
        "basename": "finetasks_arabic_ibex_fwe",
    },
}

# =============================================================================
# FineWeb2 Methodology
# =============================================================================

RANDOM_BASELINES = {
    'mmlu': 0.25,
    'exams': 0.25,
    'arc': 0.25,
    'belebele': 0.25,
    'soqal': 0.25,
    'race': 0.25,
    'sciq': 0.25,
    'xcodah': 0.25,
    'hellaswag': 0.25,
    'xnli': 0.333,
    'piqa': 0.5,
    'xstory_cloze': 0.5,
    'xcsqa': 0.2,
    'mlqa': 0.0,
    'tydiqa': 0.0,
    'arcd': 0.0,
}

ARABIC_CATEGORIES = {
    'GK': ['exams_ara_cf:_average', 'mmlu_ara_cf:_average', 'alghafa_arc_ara_cf:easy'],
    'RC': ['belebele_arb_Arab', 'soqal_ara', 'race_ar', 'sciq_ar', 'mlqa_ara', 'tydiqa_ara', 'arcd_ara'],
    'RES': ['xcodah_ara', 'alghafa_piqa_ara', 'xcsqa_ara'],
    'NLU': ['xnli2.0_ara', 'mlmm_hellaswag_ara', 'xstory_cloze_ara'],
}

ARABIC_REQUIRED_TASKS = [
    'exams_ara', 'mmlu_ara', 'alghafa_arc_ara',
    'belebele_arb_Arab', 'soqal_ara', 'race_ar', 'sciq_ar', 'mlqa_ara', 'tydiqa_ara', 'arcd_ara',
    'xcodah_ara', 'alghafa_piqa_ara', 'xcsqa_ara',
    'xnli2.0_ara', 'mlmm_hellaswag_ara', 'xstory_cloze_ara',
]

COLORS = [
    "#e91e63",  # rose
    "#009688",  # teal
    "#00bcd4",  # cyan
    "#ff9800",  # orange
    "#9c27b0",  # purple
    "#4caf50",  # green
    "#2196f3",  # blue
    "#9e9e9e",  # gray
    "#f44336",  # red
    "#795548",  # brown
    "#607d8b",  # blue-gray
    "#cddc39",  # lime
    "#ff5722",  # deep orange
]


def get_random_baseline(task_name: str) -> float:
    task_lower = task_name.lower()
    for pattern, baseline in RANDOM_BASELINES.items():
        if pattern in task_lower:
            return baseline
    return 0.25


def rescale_score(raw_score: float, task_name: str) -> float:
    baseline = get_random_baseline(task_name)
    if raw_score <= baseline:
        return 0.0
    return (raw_score - baseline) / (1 - baseline)


def extract_metric_value(task_results: Dict) -> Optional[float]:
    simple_keys = ['acc_norm', 'acc', 'acc_norm_token', 'f1', 'exact_match']
    for key in simple_keys:
        if key in task_results:
            return task_results[key]
    for key, value in task_results.items():
        if key.startswith('acc') and not key.endswith('_stderr'):
            return value
    # Check for f1_ara (Arabic F1)
    for key, value in task_results.items():
        if 'f1' in key and not key.endswith('_stderr'):
            return value
    return None


def find_result_files(step_dir: Path) -> List[Path]:
    result_files = []
    for f in step_dir.rglob("*.json"):
        if "config" in f.name.lower() or "details" in str(f):
            continue
        if "results" in f.name:
            result_files.append(f)
    return result_files


def compute_aggregate(results_dict: Dict) -> Tuple[Optional[float], Dict[str, float]]:
    category_scores = {}
    for category, task_patterns in ARABIC_CATEGORIES.items():
        scores = []
        for task_name, task_results in results_dict.items():
            if task_name == 'all':
                continue
            if any(pattern in task_name for pattern in task_patterns):
                raw = extract_metric_value(task_results)
                if raw is not None:
                    rescaled = rescale_score(raw, task_name)
                    scores.append(rescaled)
        if scores:
            category_scores[category] = np.mean(scores)

    if category_scores:
        aggregate = np.mean(list(category_scores.values()))
        return aggregate, category_scores
    return None, {}


def get_model_results(model_path: Path) -> List[Tuple[int, float, Dict[str, float]]]:
    results = []
    if not model_path.exists():
        print(f"  Warning: {model_path} does not exist")
        return results

    for step_dir in model_path.iterdir():
        if not step_dir.is_dir() or not step_dir.name.startswith("step_"):
            continue
        try:
            step = int(step_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        result_files = find_result_files(step_dir)
        if not result_files:
            continue

        all_results = {}
        for rf in result_files:
            try:
                with open(rf) as f:
                    data = json.load(f)
                if 'results' in data:
                    for task_name, task_results in data['results'].items():
                        if task_name not in all_results:
                            all_results[task_name] = task_results
            except (json.JSONDecodeError, KeyError):
                continue

        # Validate required tasks
        task_names = set(all_results.keys()) - {'all'}
        missing = [p for p in ARABIC_REQUIRED_TASKS if not any(p in t for t in task_names)]
        if missing:
            print(f"  Warning: {model_path.name} step {step} missing: {missing}")
            continue

        if all_results:
            aggregate, category_scores = compute_aggregate(all_results)
            if aggregate is not None:
                results.append((step, aggregate, category_scores))

    results.sort(key=lambda x: x[0])
    return results


def step_to_tokens_b(step: int) -> float:
    return (step * TOKENS_PER_STEP) / 1e9


def smooth_curve(x, y):
    x, y = np.array(x), np.array(y)
    if len(x) < 5:
        return x, y
    y_smooth = y.copy()
    for i in range(1, len(y) - 1):
        y_smooth[i] = 0.9 * y[i] + 0.05 * y[i-1] + 0.05 * y[i+1]
    return x, y_smooth


# =============================================================================
# Plotting
# =============================================================================

def setup_plot_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.linestyle': ':',
        'grid.color': '#cccccc',
        'grid.alpha': 0.7,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        'legend.framealpha': 1.0,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })


def create_plot(
    models_config: Dict,
    title: str,
    output_basename: str,
    smoothed: bool = True,
    legend_fontsize: int = 10,
):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    model_data = {}
    for model_name, config in models_config.items():
        # Use explicit path if provided, otherwise default to RESULTS_BASE
        if "path" in config:
            model_path = config["path"]
        else:
            model_path = RESULTS_BASE / model_name
        display_name = config["display_name"]
        print(f"  Loading {display_name}...")
        results = get_model_results(model_path)
        if results:
            model_data[display_name] = results
        else:
            print(f"    No results found")

    if not model_data:
        print(f"  No models to plot for {title}")
        plt.close(fig)
        return

    # Sort by final score descending for legend ordering
    sorted_models = sorted(
        model_data.items(),
        key=lambda x: x[1][-1][1],
        reverse=True,
    )

    BASELINE_DISPLAY = "100% AraMix-HQ (baseline)"
    color_idx = 0
    for i, (display_name, results) in enumerate(sorted_models):
        tokens = [step_to_tokens_b(r[0]) for r in results]
        scores = [r[1] for r in results]

        is_baseline = display_name == BASELINE_DISPLAY
        if is_baseline:
            color = '#333333'
            linestyle = ':'
            linewidth = 2.2
        else:
            color = COLORS[color_idx % len(COLORS)]
            color_idx += 1
            linestyle = '-'
            linewidth = 2.8

        final_score = scores[-1]
        label = f"{display_name} ({final_score:.4f})"

        if smoothed and len(tokens) >= 5:
            tokens_s, scores_s = smooth_curve(tokens, scores)
            ax.plot(tokens_s, scores_s, color=color, linewidth=linewidth,
                    linestyle=linestyle, label=label)
        else:
            ax.plot(tokens, scores, color=color, linewidth=linewidth,
                    linestyle=linestyle, label=label)

        ax.scatter([tokens[-1]], [scores[-1]], color=color, s=70, zorder=5)

        cat_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(results[-1][2].items()))
        print(f"    {display_name}: final={final_score:.4f} ({cat_str})")

    # Add vertical line at 10B tokens where base AraMix-HQ phase ends
    base_tokens_b = step_to_tokens_b(4770)  # ~10B
    ax.axvline(x=base_tokens_b, color='#888888', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.text(base_tokens_b + 0.2, ax.get_ylim()[0] + 0.003,
            '10B: base AraMix-HQ\n(all runs share this phase)',
            fontsize=8, color='#666666', va='bottom')

    ax.set_xlabel("Tokens (Billions)", fontsize=14)
    ax.set_ylabel("Aggregate Score (FineWeb2 rescaled)", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='lower right', fontsize=legend_fontsize)

    plt.tight_layout()

    pdf_path = OUTPUT_DIR / f"{output_basename}.pdf"
    png_path = OUTPUT_DIR / f"{output_basename}.png"
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    print(f"  Saved: {png_path}, {pdf_path}")
    plt.close(fig)


def generate_markdown(models_config: Dict, output_basename: str):
    lines = ["# AraTrain Arabic FineTasks Results\n"]
    lines.append("## Training Setup\n")
    lines.append("All runs share the same first 10B tokens (4770 steps) trained on 100% AraMix-HQ.")
    lines.append("From 10B to 30B tokens (steps 4770-14306), each run continues with its listed dataset mix.")
    lines.append("Total: 30B tokens per run.\n")
    lines.append("## Scoring Methodology\n")
    lines.append("FineWeb2 methodology: rescale by random baseline, avg within category, avg across categories.")
    lines.append("Categories: " + ", ".join(ARABIC_CATEGORIES.keys()) + "\n")

    all_results = {}
    for model_name, config in models_config.items():
        if "path" in config:
            model_path = config["path"]
        else:
            model_path = RESULTS_BASE / model_name
        results = get_model_results(model_path)
        if results:
            all_results[config["display_name"]] = results

    # Sort by final score
    sorted_models = sorted(all_results.items(), key=lambda x: x[1][-1][1], reverse=True)

    categories = list(ARABIC_CATEGORIES.keys())
    header = "| Model | " + " | ".join(categories) + " | **Aggregate** |"
    sep = "|---|" + "|".join(["---"] * len(categories)) + "|---|"

    lines.append("## Final Results (Step 14306 / 30B tokens)\n")
    lines.append(header)
    lines.append(sep)

    for display_name, results in sorted_models:
        _, agg, cats = results[-1]
        row = f"| {display_name} |"
        for cat in categories:
            val = cats.get(cat)
            row += f" {val:.4f} |" if val is not None else " - |"
        row += f" **{agg:.4f}** |"
        lines.append(row)

    lines.append("")
    md_path = OUTPUT_DIR / f"{output_basename}.md"
    with open(md_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"  Saved: {md_path}")


def main():
    print("=" * 60)
    print("AraTrain: Generating FineTasks Arabic Plots")
    print("(FineWeb2 methodology: rescaling + category aggregation)")
    print("=" * 60)

    for name, cfg in PLOT_CONFIGS.items():
        print(f"\n--- {name} ---")
        create_plot(
            models_config=cfg["models"],
            title=cfg["title"],
            output_basename=cfg["basename"],
        )

    print("\n--- Generating markdown summary ---")
    generate_markdown(ALL_MODELS, "finetasks_arabic_aratrain_results")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
