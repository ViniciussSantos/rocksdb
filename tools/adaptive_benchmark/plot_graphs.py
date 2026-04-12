# /// script
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "numpy",
# ]
# ///

import json
import re
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse

LABEL_MAP = {
    "ops_per_sec": "Throughput (Ops/sec)",
    "p99_latency_us": "P99 Tail Latency (μs)",
    "write_amplification": "Write Amplification Factor (WAF)",
    "io_util_avg": "Disk Utilization (%)",
    "io_write_mb_s": "Write Throughput (MB/s)",
    "compaction_cpu_sec": "Compaction CPU Time (s)",
    "throughput_per_cpu": "Efficiency (Ops/CPU-sec)",
    "alpha": "Sensitivity Parameter (α)",
    "config": "Configuration",
    "pinned": "Execution Mode",
}


def clean_label(label):
    return LABEL_MAP.get(label, label.replace("_", " ").title())


def clean_config_name(name):
    """Shortens ugly config keys for the legend."""
    name = name.replace("adaptive_", "Adpt-").replace("baseline_", "Base-")
    name = name.replace("alpha", "α=").replace("pmem", "P=").replace("cpu", "C=")
    return name


# ----------------------------
# Style (Paper-Quality)
# ----------------------------
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.4)


# ----------------------------
# Data Loading & Processing
# ----------------------------
def load_raw_df(path):
    with open(path) as f:
        data = json.load(f)

    rows = []
    for run in data["raw"]:
        config_full = run["config"]
        is_pinned = "unpinned" not in config_full.lower()
        base_config = config_full.replace("_pinned", "").replace("_unpinned", "")

        row = {
            "config_full": clean_config_name(config_full),
            "config": clean_config_name(base_config),
            "pinned": "Pinned" if is_pinned else "Unpinned",
            "run_id": run.get("run_id", 1),
        }

        row.update(run["metrics"])

        # Telemetry Parsing (Dynamic Device Detection)
        if "telemetry" in run:
            tel = run["telemetry"]
            if "cpu" in tel:
                row["cpu_avg"] = tel["cpu"].get("cpu_avg", 0)

            if "io" in tel:
                io_data = tel["io"]
                active_dev = None
                max_util = -1.0
                # Identify the disk with the highest utilization (usually the DB disk)
                for k, v in io_data.items():
                    if k.endswith("_util_avg"):
                        if v > max_util:
                            max_util = v
                            active_dev = k.replace("_util_avg", "")

                if active_dev:
                    row["io_util_avg"] = io_data.get(f"{active_dev}_util_avg", 0)
                    # Convert KB/s to MB/s for standard reporting
                    row["io_read_mb_s"] = (
                        io_data.get(f"{active_dev}_read_kb_s", 0) / 1024.0
                    )
                    row["io_write_mb_s"] = (
                        io_data.get(f"{active_dev}_write_kb_s", 0) / 1024.0
                    )
                else:
                    row["io_util_avg"], row["io_read_mb_s"], row["io_write_mb_s"] = (
                        0,
                        0,
                        0,
                    )

        rows.append(row)
    return pd.DataFrame(rows)


def add_derived_metrics(df):
    """Combines derived metrics from both previous scripts."""
    cpu_sec = df["compaction_cpu_sec"].replace(0, np.nan)
    comp_count = df["compaction_count"].replace(0, np.nan)
    gb_written = df["total_gb_written"].replace(0, np.nan)

    df["throughput_per_cpu"] = df["ops_per_sec"] / cpu_sec
    df["throughput_per_compaction"] = df["ops_per_sec"] / comp_count
    df["throughput_per_gb"] = df["ops_per_sec"] / gb_written
    df["compaction_time_ratio"] = df["compaction_time_total_sec"] / df["duration_sec"]
    return df


def extract_alpha(config_name):
    if config_name.startswith("adaptive_alpha"):
        match = re.search(r'alpha([\d\.]+)', config_name)
        if match:
            return float(match.group(1))
    elif config_name.startswith("baseline"):
        return 0.0
    return None


def prepare_sensitivity_df(df):
    sens_df = df.copy()
    sens_df["alpha"] = sens_df["config"].apply(extract_alpha)
    return sens_df[sens_df["alpha"].notna()].sort_values("alpha")


# ----------------------------
# Plotting Helpers
# ----------------------------
def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def barplot(df, metric, out, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df, x="config", y=metric, palette="viridis", errorbar=("ci", 95), ax=ax
    )
    ax.set_title(title, pad=15, fontweight='bold')
    ax.set_xlabel(clean_label("config"))
    ax.set_ylabel(clean_label(metric))
    plt.xticks(rotation=25, ha="right")
    save(fig, out)


def comparison_plot(df, metric, out, title):
    """Shows Pinned vs Unpinned side-by-side."""
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="config", y=metric, hue="pinned", errorbar=("ci", 95), ax=ax)
    ax.set_title(title, pad=15, fontweight='bold')
    ax.set_ylabel(clean_label(metric))
    ax.set_xlabel(clean_label("config"))
    plt.xticks(rotation=25, ha="right")
    save(fig, out)


def scatter_mean(df, x_metric, y_metric, out, title):
    """Clean scatterplot with no text on points and external legend."""
    mean_df = df.groupby("config")[[x_metric, y_metric]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 6))

    sns.scatterplot(
        data=mean_df,
        x=x_metric,
        y=y_metric,
        style="config",
        hue="config",
        s=150,
        palette="tab10",
        edgecolor="black",
        ax=ax,
    )

    ax.set_title(title, pad=15, fontweight='bold')
    ax.set_xlabel(clean_label(x_metric))
    ax.set_ylabel(clean_label(y_metric))
    ax.legend(
        title=clean_label("config"),
        loc='center left',
        bbox_to_anchor=(1.25, 0.5),
        ncol=1,
    )
    save(fig, out)


def normalized_bar(df, metric, baseline_name, out, title):
    # Map the baseline name to its cleaned version to find it in the current DF
    clean_baseline = clean_config_name(baseline_name)

    means = df.groupby("config")[metric].mean()
    if clean_baseline not in means.index:
        # Fallback: check if the raw name is there
        if baseline_name in means.index:
            clean_baseline = baseline_name
        else:
            return

    baseline_val = means[clean_baseline]

    # Calculate normalization
    norm_df = df.copy()
    norm_metric_name = f"norm_{metric}"
    norm_df[norm_metric_name] = norm_df[metric] / baseline_val

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting with the same professional palette
    sns.barplot(
        data=norm_df,
        x="config",
        y=norm_metric_name,
        palette="coolwarm",
        errorbar=("ci", 95),
        ax=ax,
    )

    # Clean up the visuals
    plt.axhline(1, color='red', linestyle='--', alpha=0.7, label="Baseline (1.0)")

    ax.set_title(title, pad=15, fontweight='bold')
    ax.set_xlabel(clean_label("config"))

    # Create a nice Y-axis label: "Relative Throughput (Baseline=1.0)"
    readable_metric = clean_label(metric).split('(')[0].strip()
    ax.set_ylabel(f"Relative {readable_metric} (Baseline=1.0)")

    plt.xticks(rotation=25, ha="right")

    save(fig, out)


def lineplot(df, x, y, out, title):
    fig = plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df, x=x, y=y, marker="o", errorbar=("ci", 95))
    plt.title(title)
    ax.set_xlabel(clean_label(x))
    ax.set_ylabel(clean_label(y))
    save(fig, out)


def multi_lineplot_normalized(df, x, metrics, labels, out, title):
    """Plots normalized tradeoffs for alpha sensitivity."""
    baseline = df[df["alpha"] == 0.0].groupby("alpha")[metrics].mean().iloc[0]

    mean_df = df.groupby("alpha")[metrics].mean().reset_index()
    for metric in metrics:
        mean_df[metric] = mean_df[metric] / baseline[metric]

    fig = plt.figure(figsize=(10, 6))
    for metric, label in zip(metrics, labels):
        sns.lineplot(data=mean_df, x=x, y=metric, marker="o", label=label)

    plt.axhline(1, color='black', linestyle='--', alpha=0.5)
    plt.title(title)
    plt.xlabel("Alpha (Sensitivity)")
    plt.ylabel("Relative Value (Baseline = 1.0)")
    plt.legend()
    save(fig, out)


# ----------------------------
# LLM Report Generator
# ----------------------------
def generate_llm_report(output_dir, df, captions, env_name):
    """
    Generates a markdown file pairing the graph captions with the actual
    statistical data, acting as a perfect prompt injection for an LLM.
    """
    report_path = output_dir / f"LLM_ANALYSIS_REPORT_{env_name}.md"

    with open(report_path, "w") as f:
        f.write(f"# Benchmark Analysis Context: {env_name} Environment\n\n")
        f.write(
            "This document contains figure explanations and the raw statistical data (Mean ± StdDev) backing them up. Use this data to formulate dissertation conclusions.\n\n"
        )

        for filename, data in captions.items():
            if not (output_dir / filename).exists():
                continue

            metric = data.get("metric")
            f.write(f"## Figure: {filename}\n")
            f.write(f"**Intent / Analysis:** {data['caption']}\n\n")
            f.write(f"![{filename}]({filename})\n\n")

            # Generate markdown table for the specific metric if applicable
            if metric and metric in df.columns:
                stats = df.groupby("config")[metric].agg(['mean', 'std']).reset_index()
                f.write(f"### Statistical Summary ({metric})\n")
                f.write("| Configuration | Mean | Std Dev |\n")
                f.write("|---|---|---|\n")
                for _, row in stats.iterrows():
                    f.write(
                        f"| {row['config']} | {row['mean']:.4f} | {row['std']:.4f} |\n"
                    )
                f.write("\n")
            f.write("---\n\n")


# ----------------------------
# Main Logic
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results.json")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    df = load_raw_df(args.input)
    df = add_derived_metrics(df)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    FIGURE_CAPTIONS = {
        "throughput.png": {
            "caption": "Overall throughput (operations per second) across configurations. Higher is better.",
            "metric": "ops_per_sec",
        },
        "latency_p99.png": {
            "caption": "P99 tail latency across configurations. Shows the 'stalls' experienced by users during compaction.",
            "metric": "p99_latency_us",
        },
        "waf.png": {
            "caption": "Write Amplification Factor (WAF). Lower is better. Essential for flash/PMEM longevity.",
            "metric": "write_amplification",
        },
        "compaction_cpu.png": {
            "caption": "Total CPU time spent on compaction. Lower values suggest more efficient scheduling.",
            "metric": "compaction_cpu_sec",
        },
        "compaction_count.png": {
            "caption": "Total number of compactions performed.",
            "metric": "compaction_count",
        },
        "compaction_ratio.png": {
            "caption": "Fraction of total runtime spent on compaction.",
            "metric": "compaction_time_ratio",
        },
        "cpu_efficiency.png": {
            "caption": "Throughput divided by Compaction CPU sec. Measures work per unit of energy/CPU spend.",
            "metric": "throughput_per_cpu",
        },
        "eff_compaction.png": {
            "caption": "Throughput per compaction event.",
            "metric": "throughput_per_compaction",
        },
        "tradeoff_waf_vs_perf.png": {
            "caption": "Trade-off between WAF and throughput. Ideal systems move towards the top-left (low WAF, high throughput).",
            "metric": None,
        },
        "tradeoff_latency_vs_perf.png": {
            "caption": "Trade-off between P99 latency and throughput.",
            "metric": None,
        },
        "norm_throughput.png": {
            "caption": "Throughput relative to baseline_waf. Values > 1 indicate improvement.",
            "metric": "ops_per_sec",
        },
        "norm_waf.png": {
            "caption": "WAF relative to baseline. Values < 1 indicate improvement.",
            "metric": "write_amplification",
        },
        "alpha_throughput.png": {
            "caption": "Sensitivity analysis: Total throughput as a function of α.",
            "metric": "ops_per_sec",
        },
        "alpha_waf.png": {
            "caption": "Sensitivity analysis: How increasing α reduces WAF.",
            "metric": "write_amplification",
        },
        "alpha_latency.png": {
            "caption": "Sensitivity analysis: P99 latency as a function of α.",
            "metric": "p99_latency_us",
        },
        "alpha_tradeoff.png": {
            "caption": "Relative trade-off overlay across α values. Identifies the optimal operating region.",
            "metric": None,
        },
        "bias_comparison_throughput.png": {
            "caption": "Direct comparison of Pinned vs Unpinned environments. Highlights scheduler bias and variance.",
            "metric": "ops_per_sec",
        },
        "bias_comparison_latency.png": {
            "caption": "Direct comparison of tail latency across environments.",
            "metric": "p99_latency_us",
        },
        "io_utilization.png": {
            "caption": "Average Disk Utilization percentage. High values indicate IO bottlenecks.",
            "metric": "io_util_avg",
        },
        "io_write_throughput.png": {
            "caption": "Disk Write throughput in MB/s. Shows physical write pressure.",
            "metric": "io_write_mb_s",
        },
    }

    # 1. Process Environments Separately
    for mode in ["Pinned", "Unpinned"]:
        mode_dir = root / f"{mode.lower()}_env"
        mode_dir.mkdir(parents=True, exist_ok=True)
        m_df = df[df["pinned"] == mode]

        if m_df.empty:
            continue

        # Standard Performance & IO
        barplot(
            m_df,
            "io_util_avg",
            mode_dir / "io_utilization.png",
            f"Disk Utilization ({mode})",
        )
        barplot(
            m_df,
            "io_write_mb_s",
            mode_dir / "io_write_throughput.png",
            f"Write Throughput ({mode})",
        )

        # Bar plots
        barplot(
            m_df,
            "ops_per_sec",
            mode_dir / "throughput.png",
            f"Throughput ({mode})",
        )
        barplot(
            m_df,
            "p99_latency_us",
            mode_dir / "latency_p99.png",
            f"Tail Latency ({mode})",
        )
        barplot(m_df, "write_amplification", mode_dir / "waf.png", f"WAF ({mode})")
        barplot(
            m_df,
            "compaction_cpu_sec",
            mode_dir / "compaction_cpu.png",
            f"Compaction CPU Time ({mode})",
        )
        barplot(
            m_df,
            "compaction_count",
            mode_dir / "compaction_count.png",
            f"Compaction Count ({mode})",
        )
        barplot(
            m_df,
            "compaction_time_ratio",
            mode_dir / "compaction_ratio.png",
            f"Compaction Time Ratio ({mode})",
        )

        # Efficiency
        barplot(
            m_df,
            "throughput_per_cpu",
            mode_dir / "cpu_efficiency.png",
            f"CPU Efficiency ({mode})",
        )
        barplot(
            m_df,
            "throughput_per_compaction",
            mode_dir / "eff_compaction.png",
            f"Compaction Efficiency ({mode})",
        )

        # Tradeoff Scatters
        scatter_mean(
            m_df,
            "write_amplification",
            "ops_per_sec",
            mode_dir / "tradeoff_waf_vs_perf.png",
            f"WAF vs Throughput ({mode})",
        )
        scatter_mean(
            m_df,
            "p99_latency_us",
            "ops_per_sec",
            mode_dir / "tradeoff_latency_vs_perf.png",
            f"Latency vs Throughput ({mode})",
        )

        # Normalized Bars
        normalized_bar(
            m_df,
            "ops_per_sec",
            "baseline_waf",
            mode_dir / "norm_throughput.png",
            f"Relative Throughput ({mode})",
        )
        normalized_bar(
            m_df,
            "write_amplification",
            "baseline_waf",
            mode_dir / "norm_waf.png",
            f"Relative WAF ({mode})",
        )

        # Sensitivity Analysis
        s_df = prepare_sensitivity_df(m_df)
        if not s_df.empty:
            lineplot(
                s_df,
                "alpha",
                "ops_per_sec",
                mode_dir / "alpha_throughput.png",
                f"Throughput Sensitivity ({mode})",
            )
            lineplot(
                s_df,
                "alpha",
                "write_amplification",
                mode_dir / "alpha_waf.png",
                f"WAF Sensitivity ({mode})",
            )
            lineplot(
                s_df,
                "alpha",
                "p99_latency_us",
                mode_dir / "alpha_latency.png",
                f"Latency Sensitivity ({mode})",
            )

            multi_lineplot_normalized(
                s_df,
                "alpha",
                ["ops_per_sec", "write_amplification", "p99_latency_us"],
                ["Throughput", "WAF", "P99 Latency"],
                mode_dir / "alpha_tradeoff.png",
                f"Relative Tradeoff Dynamics ({mode})",
            )

        # Generate the LLM Context Report for this environment
        generate_llm_report(mode_dir, m_df, FIGURE_CAPTIONS, mode)

    # 2. Global Bias Analysis (Cross-Environment)
    bias_dir = root / "bias_analysis"
    bias_dir.mkdir(parents=True, exist_ok=True)
    comparison_plot(
        df,
        "ops_per_sec",
        bias_dir / "bias_comparison_throughput.png",
        "Scheduling Impact: Throughput",
    )
    comparison_plot(
        df,
        "p99_latency_us",
        bias_dir / "bias_comparison_latency.png",
        "Scheduling Impact: Tail Latency",
    )

    # Generate an LLM report specifically for the bias analysis
    generate_llm_report(bias_dir, df, FIGURE_CAPTIONS, "Global_Bias_Comparison")

    print(f"All graphs and LLM reports successfully saved to {root}")


if __name__ == "__main__":
    main()
