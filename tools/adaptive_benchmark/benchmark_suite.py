#!/usr/bin/env python3

import subprocess
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import argparse
import re


def aggregate_results(results):
    grouped = {}

    for r in results:
        key = r["config"]
        grouped.setdefault(key, []).append(r["metrics"])

    aggregated = {}

    for key, runs in grouped.items():
        metrics = {}

        for metric in runs[0].keys():
            values = [r[metric] for r in runs if r[metric] > 0]

            if not values:
                continue

            arr = np.array(values)

            metrics[metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "ci95": float(1.96 * np.std(arr) / np.sqrt(len(arr))),
            }

        aggregated[key] = metrics

    return aggregated


@dataclass
class BenchmarkConfig:
    name: str
    num_operations: int
    value_size: int
    threads: int
    adaptive_enabled: bool

    adaptive_sensitivity: float = 2.0
    adaptive_pmem_weight: float = 0.6
    adaptive_cpu_weight: float = 0.4

    max_compaction_threads: int = 8
    adaptive_min_threads: int = 1
    level0_trigger: int = 4
    write_buffer_size: int = 64 * 1024 * 1024

    def validate(self):
        if self.adaptive_enabled:
            weight_sum = self.adaptive_pmem_weight + self.adaptive_cpu_weight
            if abs(weight_sum - 1.0) > 1e-6:
                raise ValueError(f"Invalid weights: sum={weight_sum}")

    def to_db_bench_args(self, db_path: str) -> List[str]:
        self.validate()

        args = [
            f"--db={db_path}",
            f"--num={self.num_operations}",
            f"--value_size={self.value_size}",
            f"--threads={self.threads}",
            f"--max_background_compactions={self.max_compaction_threads}",
            f"--level0_file_num_compaction_trigger={self.level0_trigger}",
            f"--write_buffer_size={self.write_buffer_size}",
            "--statistics=true",
            "--histogram=true",
        ]

        args.append(
            f"--enable_adaptive_compaction={str(self.adaptive_enabled).lower()}"
        )

        if self.adaptive_enabled:
            args.extend(
                [
                    f"--adaptive_compaction_sensitivity={self.adaptive_sensitivity}",
                    f"--adaptive_pmem_weight={self.adaptive_pmem_weight}",
                    f"--adaptive_cpu_weight={self.adaptive_cpu_weight}",
                    "--adaptive_critical_threshold=5.0",
                    "--adaptive_max_deferrals=10",
                ]
            )

        return args


class BenchmarkRunner:
    def __init__(
        self,
        db_bench_path: str,
        output_dir: str,
        dry_run=False,
        cpu_cores="0-3",
        warmup_ops=100_000,
        enable_telemetry=True,
    ):

        self.db_bench_path = db_bench_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dry_run = dry_run
        self.cpu_cores = cpu_cores
        self.warmup_ops = warmup_ops
        self.enable_telemetry = enable_telemetry

    def run_benchmark(self, config: BenchmarkConfig, benchmark_type="fillrandom"):

        print(f"\n=== Running {config.name} ===")

        db_path = self.output_dir / f"db_{config.name}"
        if db_path.exists():
            subprocess.run(["rm", "-rf", str(db_path)])
        db_path.mkdir(parents=True)

        # --- Warmup ---
        self._warmup(db_path)

        # --- Actual run ---
        cmd = self._build_cmd(config, benchmark_type, db_path)

        if self.dry_run:
            print("DRY RUN:", " ".join(cmd))
            return None

        output_file = self.output_dir / f"{config.name}.txt"

        telemetry = {}
        telemetry_proc = None

        if self.enable_telemetry:
            telemetry_proc = self._start_telemetry(config.name)

        start = time.time()

        with open(output_file, "w") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            proc.wait()

        duration = time.time() - start

        if telemetry_proc:
            telemetry = self._stop_telemetry(telemetry_proc)

        metrics = self._parse_output(output_file)
        metrics["duration_sec"] = duration

        return {"config": config.name, "metrics": metrics, "telemetry": telemetry}

    def _warmup(self, db_path: Path):
        print("-> Warmup phase")

        cmd = [
            "taskset",
            "-c",
            self.cpu_cores,
            self.db_bench_path,
            f"--db={db_path}",
            "--benchmarks=fillrandom",
            f"--num={self.warmup_ops}",
            "--threads=1",
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _build_cmd(self, config, benchmark_type, db_path):

        cmd = [
            "taskset",
            "-c",
            self.cpu_cores,
            self.db_bench_path,
            f"--benchmarks={benchmark_type},stats",
            "--statistics",
        ]

        cmd.extend(config.to_db_bench_args(str(db_path)))
        return cmd

    def _start_telemetry(self, name):

        pidstat_file = self.output_dir / f"{name}_pidstat.log"
        iostat_file = self.output_dir / f"{name}_iostat.log"

        pidstat = subprocess.Popen(
            ["pidstat", "-dur", "1"],
            stdout=open(pidstat_file, "w"),
            stderr=subprocess.DEVNULL,
        )

        iostat = subprocess.Popen(
            ["iostat", "-dx", "1"],
            stdout=open(iostat_file, "w"),
            stderr=subprocess.DEVNULL,
        )

        return (pidstat, iostat, pidstat_file, iostat_file)

    def _stop_telemetry(self, procs):

        pidstat, iostat, pid_file, io_file = procs

        pidstat.terminate()
        iostat.terminate()

        # Give processes a moment to flush
        time.sleep(1)

        cpu_stats = self._parse_pidstat(pid_file)
        io_stats = self._parse_iostat(io_file)

        return {"cpu": cpu_stats, "io": io_stats}

    def _parse_output(self, file: Path):

        text = file.read_text()

        def extract(pattern):
            m = re.search(pattern, text, re.IGNORECASE)
            return float(m.group(1)) if m else 0.0

        return {
            "ops_per_sec": extract(r"([\d\.eE\+\-]+)\s+ops/sec"),
            "p50_latency_us": extract(r"P50.*?:\s+([\d\.]+)"),
            "p95_latency_us": extract(r"P95.*?:\s+([\d\.]+)"),
            "p99_latency_us": extract(r"P99[^\.].*?:\s+([\d\.]+)"),
            "p999_latency_us": extract(r"P99\.9.*?:\s+([\d\.]+)"),
            "avg_latency_us": extract(r"Average.*?:\s+([\d\.]+)"),
            "write_amplification": extract(r"write amplification.*?([\d\.]+)"),
        }

    def _parse_pidstat(self, file: Path):
        cpu_values = []

        with open(file) as f:
            for line in f:
                parts = line.split()

                # Typical pidstat format:
                # time UID PID %usr %system %CPU ...
                if len(parts) > 7 and parts[0].count(":") == 2:
                    try:
                        cpu = float(parts[7])  # %CPU column
                        cpu_values.append(cpu)
                    except:  # noqa: E722
                        continue

        if not cpu_values:
            return {}

        return {
            "cpu_avg": sum(cpu_values) / len(cpu_values),
            "cpu_max": max(cpu_values),
        }

    def _parse_iostat(self, file: Path):
        util_values = []
        read_kb = []
        write_kb = []

        with open(file) as f:
            for line in f:
                parts = line.split()

                # Linux iostat -dx format
                if (
                    len(parts) >= 14
                    and parts[0].startswith("sd")
                    or parts[0].startswith("nvme")
                ):
                    try:
                        read_kb.append(float(parts[5]))  # rKB/s
                        write_kb.append(float(parts[6]))  # wKB/s
                        util_values.append(float(parts[-1]))  # %util
                    except:
                        continue

        if not util_values:
            return {}

        return {
            "read_kb_s_avg": sum(read_kb) / len(read_kb),
            "write_kb_s_avg": sum(write_kb) / len(write_kb),
            "util_avg": sum(util_values) / len(util_values),
            "util_max": max(util_values),
        }


class ExperimentSuite:
    def __init__(self, runner: BenchmarkRunner, repeat: int):
        self.runner = runner
        self.repeat = repeat
        self.results = []

    # ------------------------------------------------------------
    # Core execution helper
    # ------------------------------------------------------------
    def run_config(self, config, benchmark):
        for i in range(self.repeat):
            print(f"\nRun {i + 1}/{self.repeat} - {config.name}")

            result = self.runner.run_benchmark(config, benchmark)

            if result:
                result["run_id"] = i + 1
                self.results.append(result)

    # ------------------------------------------------------------
    # Experiment 1: Write Amplification
    # ------------------------------------------------------------
    def run_waf(self):
        print("\n=== Experiment: Write Amplification ===")

        configs = [
            BenchmarkConfig("baseline_waf", 1_000_000, 1000, 1, False),
            BenchmarkConfig(
                "adaptive_alpha1", 1_000_000, 1000, 1, True, adaptive_sensitivity=1.0
            ),
            BenchmarkConfig(
                "adaptive_alpha2", 1_000_000, 1000, 1, True, adaptive_sensitivity=2.0
            ),
            BenchmarkConfig(
                "adaptive_alpha3", 1_000_000, 1000, 1, True, adaptive_sensitivity=3.0
            ),
        ]

        for c in configs:
            self.run_config(c, "fillrandom")

    # ------------------------------------------------------------
    # Experiment 2: Latency under load
    # ------------------------------------------------------------
    def run_latency(self):
        print("\n=== Experiment: Latency ===")

        configs = [
            BenchmarkConfig("baseline_latency", 500_000, 2000, 8, False),
            BenchmarkConfig("adaptive_latency", 500_000, 2000, 8, True),
        ]

        for c in configs:
            self.run_config(c, "fillrandom")

    # ------------------------------------------------------------
    # Experiment 3: Thread scaling
    # ------------------------------------------------------------
    def run_scaling(self):
        print("\n=== Experiment: Scaling ===")

        for threads in [1, 2, 4, 8, 16]:
            baseline = BenchmarkConfig(
                f"baseline_scaling_{threads}",
                300_000,
                1000,
                threads,
                False,
            )

            adaptive = BenchmarkConfig(
                f"adaptive_scaling_{threads}",
                300_000,
                1000,
                threads,
                True,
            )

            self.run_config(baseline, "fillrandom")
            self.run_config(adaptive, "fillrandom")

    # ------------------------------------------------------------
    # Experiment 4: Mixed workload
    # ------------------------------------------------------------
    def run_mixed(self):
        print("\n=== Experiment: Mixed Workload ===")

        configs = [
            BenchmarkConfig("baseline_mixed", 1_000_000, 1000, 8, False),
            BenchmarkConfig("adaptive_mixed", 1_000_000, 1000, 8, True),
        ]

        for c in configs:
            self.run_config(c, "readrandomwriterandom")

    # ------------------------------------------------------------
    # Experiment 5: Burst workload
    # ------------------------------------------------------------
    def run_burst(self):
        print("\n=== Experiment: Burst ===")

        configs = [
            BenchmarkConfig("baseline_burst", 800_000, 1500, 16, False),
            BenchmarkConfig(
                "adaptive_burst", 800_000, 1500, 16, True, adaptive_sensitivity=2.5
            ),
        ]

        for c in configs:
            self.run_config(c, "overwrite")

    # ------------------------------------------------------------
    # Run all experiments
    # ------------------------------------------------------------
    def run_all(self):
        self.run_waf()
        self.run_latency()
        self.run_scaling()
        self.run_mixed()
        self.run_burst()

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------
    def save(self, path):

        aggregated = aggregate_results(self.results)

        with open(path, "w") as f:
            json.dump({"raw": self.results, "aggregated": aggregated}, f, indent=2)

        print(f"\nSaved results to {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Adaptive Compaction Scheduler Benchmark Suite'
    )
    parser.add_argument("--db-bench", required=True, help="Path to db_bench binary")
    parser.add_argument(
        "--output-dir", default="./results", help="Output directory for results"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat each experiment",
    )
    parser.add_argument(
        "--experiments",
        nargs='+',
        choices=['waf', 'latency', 'scaling', 'mixed', 'burst', 'all'],
        default=['all'],
        help="Which experiments to run",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run with reduced operation counts for quick testing",
    )
    parser.add_argument(
        "--cpu-cores", default="0-3", help="CPU cores to pin processes to (e.g., '0-7')"
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Disable CPU/IO telemetry collection",
    )
    parser.add_argument(
        "--warmup-ops", type=int, default=100_000, help="Number of warmup operations"
    )

    args = parser.parse_args()

    runner = BenchmarkRunner(
        args.db_bench,
        args.output_dir,
        dry_run=args.dry_run,
        cpu_cores=args.cpu_cores,
        warmup_ops=args.warmup_ops,
        enable_telemetry=not args.no_telemetry,
    )
    suite = ExperimentSuite(runner, args.repeat)

    # Map experiment names to methods
    experiment_map = {
        'waf': suite.run_waf,
        'latency': suite.run_latency,
        'scaling': suite.run_scaling,
        'mixed': suite.run_mixed,
        'burst': suite.run_burst,
    }

    # Determine which experiments to run
    if 'all' in args.experiments:
        experiments_to_run = experiment_map.keys()
    else:
        experiments_to_run = args.experiments

    # Run selected experiments
    print(f"\n{'=' * 80}")
    print(f"Starting Benchmark Suite")
    print(f"Experiments: {', '.join(experiments_to_run)}")
    print(f"Repeats: {args.repeat}")
    print(f"{'=' * 80}\n")

    for exp_name in experiments_to_run:
        experiment_map[exp_name]()

    # Save results
    if not args.dry_run:
        suite.save(Path(args.output_dir) / "results.json")


if __name__ == "__main__":
    main()
