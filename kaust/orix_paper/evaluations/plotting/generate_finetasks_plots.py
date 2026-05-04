#!/usr/bin/env python
"""Generate FineTasks plots for the orix NeurIPS bundle.

Reads:
  - kaust/orix_paper/config/evaluation/finetasks_metric_spec.json
      Per-language task → category, metric, random baseline.
  - kaust/orix_paper/config/runs_metadata.json
      Per-run language, family, kind (new/prior), results subpath.

Walks ~/nanotron-orix-neurips-results-260430/results/<run>/step_*/results_*.json
(or any *_gen* sibling files for generative tasks) and computes per-checkpoint
aggregate scores using the FineTasks protocol:

  rescaled = max(0, (score - baseline) / (1 - baseline))   # per-task
  cat_score = mean(rescaled across tasks in category)        # per-category
  aggregate = mean(cat_score across categories)              # final

Aggregates seed-runs into families (mean line + ±1 stdev shaded band) and
emits per-language PDF/PNG/MD outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path("/home/alrashsm/github/nanotron")
CONFIG_DIR = REPO_ROOT / "kaust/orix_paper/config"
DEFAULT_METRIC_SPEC = CONFIG_DIR / "evaluation/finetasks_metric_spec.json"
DEFAULT_RUNS_META = CONFIG_DIR / "runs_metadata.json"
DEFAULT_RESULTS_BASE = Path.home() / "nanotron-orix-neurips-results-260430/results"
DEFAULT_OUTPUT_DIR = Path.home() / "nanotron-orix-neurips-results-260430/figures"

# Lighteval JSON key per metric-spec name.
METRIC_KEY_MAP = {
    "acc_pmi":   "acc_norm_pmi",
    "acc_token": "acc_norm_token",
    "acc_char":  "acc_norm",
    "acc":       "acc",
    "f1":        "f1",
}

# Fallback chain when the preferred key is missing (lighteval sometimes only
# emits a subset depending on task type). We log what was actually used.
METRIC_FALLBACKS = {
    "acc_pmi":   ["acc_norm_pmi", "acc_norm_token", "acc_norm", "acc"],
    "acc_token": ["acc_norm_token", "acc_norm", "acc"],
    "acc_char":  ["acc_norm", "acc"],
    "acc":       ["acc", "acc_norm"],
    "f1":        ["f1", "exact_match"],
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_metric_spec(path: Path = DEFAULT_METRIC_SPEC) -> dict:
    with open(path) as f:
        return json.load(f)


def load_runs_metadata(path: Path = DEFAULT_RUNS_META) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Lighteval result parsing
# ---------------------------------------------------------------------------

def find_results_files(step_dir: Path) -> list[Path]:
    """Find all results_*.json files under a step_<N> directory (recursive)."""
    return [p for p in step_dir.rglob("results_*.json") if "details" not in p.parts]


def merge_step_results(step_dir: Path) -> dict:
    """Merge all results JSONs under one step_<N> dir into a single results dict
    keyed by full task name."""
    merged: dict = {}
    for f in find_results_files(step_dir):
        try:
            with open(f) as h:
                data = json.load(h)
        except (json.JSONDecodeError, OSError):
            continue
        for tname, tres in (data.get("results") or {}).items():
            if tname == "all":
                continue
            merged.setdefault(tname, tres)
    return merged


def task_matches(spec_task: str, lighteval_task: str) -> bool:
    """Match a spec task pattern against a lighteval task name.

    Lighteval names look like: 'lighteval|<spec_task>|0' or 'lighteval|<spec_task>|5'.
    For tasks with subjects (e.g., meta_mmlu_ita_cf:abstract_algebra), the spec
    can pin a single subject (exact match) or just the family ('meta_mmlu_ita_cf')
    to mean "average across subjects" (handled by caller).
    """
    # Strip 'lighteval|' prefix and trailing |<n_shots>
    if "|" in lighteval_task:
        parts = lighteval_task.split("|")
        if len(parts) >= 3:
            core = parts[1]
        else:
            core = parts[0]
    else:
        core = lighteval_task
    return core == spec_task


def task_family_of(lighteval_task: str) -> str:
    """For 'lighteval|meta_mmlu_ita_cf:abstract_algebra|0' -> 'meta_mmlu_ita_cf'."""
    if "|" in lighteval_task:
        parts = lighteval_task.split("|")
        core = parts[1] if len(parts) >= 3 else parts[0]
    else:
        core = lighteval_task
    return core.split(":")[0]


def extract_metric(task_results: dict, metric_spec: str) -> Optional[float]:
    """Extract value from a lighteval task results dict using fallback order."""
    for key in METRIC_FALLBACKS.get(metric_spec, [METRIC_KEY_MAP[metric_spec]]):
        if key in task_results and task_results[key] is not None:
            return float(task_results[key])
    # Some lighteval keys are formatted as "acc:logprob_normalization=..." etc.
    for k, v in task_results.items():
        if k.startswith("acc") and not k.endswith("_stderr") and v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def rescale(score: float, baseline: float) -> float:
    if score <= baseline:
        return 0.0
    if baseline >= 1.0:
        return 0.0
    return (score - baseline) / (1.0 - baseline)


# ---------------------------------------------------------------------------
# Per-language aggregator
# ---------------------------------------------------------------------------

def aggregate_step(
    merged_results: dict,
    lang_spec: list[dict],
) -> tuple[Optional[float], dict[str, float], dict[str, float]]:
    """Compute (aggregate, category_scores, per_task_scores) for one (run, step).

    For tasks with subjects_average=True, averages all matching subjects first,
    then treats the result as one task in its category.
    """
    by_cat: dict[str, list[float]] = defaultdict(list)
    per_task: dict[str, float] = {}

    for spec in lang_spec:
        spec_task = spec["task"]
        category = spec["category"]
        metric = spec["metric"]
        baseline = spec["baseline"]
        avg_subjects = spec.get("subjects_average", False)

        matched_scores: list[float] = []
        for ltask, lres in merged_results.items():
            if avg_subjects:
                if task_family_of(ltask) != spec_task:
                    continue
            else:
                if not task_matches(spec_task, ltask):
                    continue
            v = extract_metric(lres, metric)
            if v is not None:
                matched_scores.append(v)
        if not matched_scores:
            continue
        raw = float(np.mean(matched_scores))
        rescaled = rescale(raw, baseline)
        per_task[spec_task] = rescaled
        by_cat[category].append(rescaled)

    if not by_cat:
        return None, {}, per_task

    cat_scores = {c: float(np.mean(v)) for c, v in by_cat.items()}
    aggregate = float(np.mean(list(cat_scores.values())))
    return aggregate, cat_scores, per_task


def collect_run_results(run_path: Path, lang_spec: list[dict]) -> list[dict]:
    """Walk run_path/step_*/, return list of dicts {step, tokens_b, aggregate,
    category_scores, per_task_scores} sorted by step."""
    out: list[dict] = []
    if not run_path.exists():
        return out
    for step_dir in run_path.iterdir():
        if not step_dir.is_dir() or not step_dir.name.startswith("step_"):
            continue
        try:
            step = int(step_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        merged = merge_step_results(step_dir)
        if not merged:
            continue
        agg, cat, ptask = aggregate_step(merged, lang_spec)
        if agg is None:
            continue
        out.append({
            "step": step,
            "aggregate": agg,
            "categories": cat,
            "per_task": ptask,
        })
    out.sort(key=lambda r: r["step"])
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 1.2,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.color": "#cccccc",
        "grid.alpha": 0.7,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def family_curves(
    runs_meta: dict,
    lang: str,
    lang_spec: list[dict],
    results_base: Path,
    tokens_per_step: int,
) -> dict[str, dict]:
    """Return {family_name: {tokens, mean, std, per_run: [(run, [(step, agg)...]), ...], color}}.

    Combines all runs (NEW + PRIOR via symlink) for each family, computes per-step
    mean and stdev across runs. Steps are aligned by their actual int value;
    families with multiple seeds get a shaded band for stdev > 0.
    """
    families = runs_meta["families"].get(lang, {})
    runs_for_lang = {
        rname: rmeta for rname, rmeta in runs_meta["runs"].items()
        if rmeta["language"] == lang
    }

    by_family: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for rname, rmeta in runs_for_lang.items():
        family = rmeta["family"]
        run_path = results_base / rmeta["results_subpath"]
        run_results = collect_run_results(run_path, lang_spec)
        if run_results:
            by_family[family].append((rname, run_results))

    output: dict[str, dict] = {}
    for family, runs_data in by_family.items():
        # Collect all unique steps across runs in this family
        all_steps = sorted({r["step"] for _, run_results in runs_data for r in run_results})
        mean: list[float] = []
        std: list[float] = []
        tokens: list[float] = []
        for step in all_steps:
            vals = []
            for _, run_results in runs_data:
                for r in run_results:
                    if r["step"] == step:
                        vals.append(r["aggregate"])
                        break
            if not vals:
                continue
            tokens.append(step * tokens_per_step / 1e9)
            mean.append(float(np.mean(vals)))
            std.append(float(np.std(vals)) if len(vals) > 1 else 0.0)

        meta = families.get(family, {})
        output[family] = {
            "tokens": tokens,
            "mean": mean,
            "std": std,
            "color": meta.get("color"),
            "source": meta.get("source", ""),
            "runs": [r for r, _ in runs_data],
            "n_runs": len(runs_data),
            "per_run": runs_data,
        }
    return output


def plot_language(
    lang: str,
    curves: dict[str, dict],
    output_dir: Path,
    title_extra: str = "",
):
    if not curves:
        print(f"[{lang}] no data to plot")
        return
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Order families by final aggregate score, descending
    ordered = sorted(curves.items(), key=lambda kv: -(kv[1]["mean"][-1] if kv[1]["mean"] else 0))
    for family, c in ordered:
        if not c["tokens"]:
            continue
        color = c["color"] or None
        x = np.array(c["tokens"])
        y = np.array(c["mean"])
        s = np.array(c["std"])
        label = family if c["n_runs"] == 1 else f"{family} (n={c['n_runs']})"
        ax.plot(x, y, color=color, linewidth=2.6, label=label)
        if (s > 0).any():
            ax.fill_between(x, y - s, y + s, color=color, alpha=0.18, linewidth=0)
        ax.scatter([x[-1]], [y[-1]], color=color, s=70, zorder=5)

    ax.set_xlabel("Tokens (Billions)", fontsize=14)
    ax.set_ylabel("FineTasks Aggregate Score", fontsize=14)
    ax.set_title(f"FineTasks — {lang.title()}{title_extra}", fontsize=15, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"finetasks_{lang}_results"
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[{lang}] wrote {base}.{{pdf,png}}")


def plot_language_per_category(
    lang: str,
    runs_meta: dict,
    lang_spec: list[dict],
    results_base: Path,
    tokens_per_step: int,
    output_dir: Path,
):
    """One subplot per category, per family curve."""
    families_meta = runs_meta["families"].get(lang, {})
    runs_for_lang = {
        rname: rmeta for rname, rmeta in runs_meta["runs"].items()
        if rmeta["language"] == lang
    }
    cats = sorted({s["category"] for s in lang_spec})
    if not cats:
        return

    # Collect per-family per-cat curves
    fam_cat: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    fam_cat_runs: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for rname, rmeta in runs_for_lang.items():
        run_path = results_base / rmeta["results_subpath"]
        run_results = collect_run_results(run_path, lang_spec)
        for r in run_results:
            for cat, val in r["categories"].items():
                fam_cat_runs[rmeta["family"]][cat].append((r["step"], val))

    # Aggregate by family per category
    per_fam_cat: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(dict))
    for family, cmap in fam_cat_runs.items():
        for cat, items in cmap.items():
            by_step: dict[int, list[float]] = defaultdict(list)
            for step, v in items:
                by_step[step].append(v)
            steps = sorted(by_step.keys())
            tokens = [s * tokens_per_step / 1e9 for s in steps]
            mean = [float(np.mean(by_step[s])) for s in steps]
            std = [float(np.std(by_step[s])) if len(by_step[s]) > 1 else 0.0 for s in steps]
            per_fam_cat[family][cat] = {"tokens": tokens, "mean": mean, "std": std}

    if not per_fam_cat:
        return

    setup_style()
    n_cats = len(cats)
    fig, axes = plt.subplots(1, n_cats, figsize=(5 * n_cats, 5), sharey=False)
    if n_cats == 1:
        axes = [axes]
    for ax, cat in zip(axes, cats):
        for family, by_cat in per_fam_cat.items():
            if cat not in by_cat:
                continue
            d = by_cat[cat]
            color = families_meta.get(family, {}).get("color")
            x = np.array(d["tokens"])
            y = np.array(d["mean"])
            s = np.array(d["std"])
            ax.plot(x, y, color=color, linewidth=2.2, label=family)
            if (s > 0).any():
                ax.fill_between(x, y - s, y + s, color=color, alpha=0.18, linewidth=0)
        ax.set_title(cat)
        ax.set_xlabel("Tokens (B)")
        ax.grid(True, linestyle=":", alpha=0.7)
    axes[0].set_ylabel("Rescaled Score")
    axes[-1].legend(loc="best", fontsize=9)
    fig.suptitle(f"FineTasks — {lang.title()} (per-category)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"finetasks_{lang}_categories"
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[{lang}] wrote {base}.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Markdown summaries
# ---------------------------------------------------------------------------

def write_language_md(
    lang: str,
    curves: dict[str, dict],
    runs_meta: dict,
    lang_spec: list[dict],
    output_dir: Path,
):
    if not curves:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    md = output_dir / f"finetasks_{lang}_results.md"
    families_meta = runs_meta["families"].get(lang, {})
    cats = sorted({s["category"] for s in lang_spec})

    lines = [
        f"# FineTasks {lang.title()} Results",
        "",
        "## Methodology",
        "",
        "FineTasks aggregation per FineWeb-2 (arXiv 2506.20920):",
        "- per-task: rescaled = max(0, (raw - baseline) / (1 - baseline))",
        "- per-category: mean of rescaled task scores in {GK, RC, RES, NLU}",
        "- aggregate: macro-mean across categories",
        "",
        "Per-task metric (acc_pmi / acc_token / acc_char / f1) follows the paper's",
        "Tables 35–39 (canary languages from `lighteval/examples/tasks/fine_tasks/`,",
        "unseen languages from Tables 36–38).",
        "",
        "## Source datasets per family",
        "",
        "| Family | Source | Color | n_runs |",
        "| --- | --- | --- | --- |",
    ]
    for family, c in curves.items():
        meta = families_meta.get(family, {})
        lines.append(
            f"| {family} | {meta.get('source','—')} | `{meta.get('color','—')}` | {c['n_runs']} |"
        )

    # Final-step table
    lines += ["", "## Final-step Aggregate (highest available step)", "",
              "| Family | Tokens (B) | Aggregate | " + " | ".join(cats) + " |",
              "|" + "|".join(["---"] * (len(cats) + 3)) + "|"]
    for family, c in sorted(curves.items(), key=lambda kv: -(kv[1]["mean"][-1] if kv[1]["mean"] else 0)):
        if not c["mean"]:
            continue
        # We need the final-step category breakdown; recompute from per-run results
        cat_means: dict[str, list[float]] = defaultdict(list)
        for rname, run_results in c["per_run"]:
            if not run_results:
                continue
            last = run_results[-1]
            for cat, v in last["categories"].items():
                cat_means[cat].append(v)
        cat_str = " | ".join(f"{np.mean(cat_means[cat]):.4f}" if cat_means.get(cat) else "—" for cat in cats)
        lines.append(f"| {family} | {c['tokens'][-1]:.2f} | {c['mean'][-1]:.4f} | {cat_str} |")

    md.write_text("\n".join(lines) + "\n")
    print(f"[{lang}] wrote {md}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-spec", type=Path, default=DEFAULT_METRIC_SPEC)
    parser.add_argument("--runs-meta", type=Path, default=DEFAULT_RUNS_META)
    parser.add_argument("--results-base", type=Path, default=DEFAULT_RESULTS_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--languages", nargs="*", default=None,
                        help="Subset of languages; default: all in metric spec with available runs")
    args = parser.parse_args()

    spec = load_metric_spec(args.metric_spec)
    runs_meta = load_runs_metadata(args.runs_meta)
    tokens_per_step = spec.get("tokens_per_step", 2_097_152)

    languages = args.languages or list(spec["languages"].keys())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for lang in languages:
        if lang not in spec["languages"]:
            print(f"[{lang}] not in metric spec — skipping")
            continue
        lang_spec = spec["languages"][lang]["tasks"]
        curves = family_curves(runs_meta, lang, lang_spec, args.results_base, tokens_per_step)
        if not curves:
            print(f"[{lang}] no result data — skipping plots")
            continue
        plot_language(lang, curves, args.output_dir)
        plot_language_per_category(
            lang, runs_meta, lang_spec, args.results_base, tokens_per_step, args.output_dir
        )
        write_language_md(lang, curves, runs_meta, lang_spec, args.output_dir)
        summary[lang] = {
            family: {
                "n_runs": c["n_runs"],
                "runs": c["runs"],
                "final_tokens_b": (c["tokens"][-1] if c["tokens"] else None),
                "final_aggregate": (c["mean"][-1] if c["mean"] else None),
                "final_aggregate_std": (c["std"][-1] if c["std"] else None),
            }
            for family, c in curves.items()
        }

    summary_path = args.output_dir / "finetasks_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary {summary_path}")


if __name__ == "__main__":
    main()
