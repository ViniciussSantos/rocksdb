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
            f"--benchmarks={benchmark_type}",
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

        return {"pidstat_file": str(pid_file), "iostat_file": str(io_file)}

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


class ExperimentSuite:
    def __init__(self, runner: BenchmarkRunner, repeat: int):
        self.runner = runner
        self.repeat = repeat
        self.results = []

    def run_config(self, config, benchmark):
        for i in range(self.repeat):
            print(f"\nRun {i + 1}/{self.repeat} - {config.name}")
            result = self.runner.run_benchmark(config, benchmark)
            if result:
                self.results.append(result)

    def run_waf(self):
        configs = [
            BenchmarkConfig("baseline_10M", 1_000_000, 1000, 1, False),
            BenchmarkConfig("adaptive_alpha2", 1_000_000, 1000, 1, True),
        ]
        for c in configs:
            self.run_config(c, "fillrandom")

    def save(self, path):
        aggregated = aggregate_results(self.results)

        with open(path, "w") as f:
            json.dump({"raw": self.results, "aggregated": aggregated}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-bench", required=True)
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--cpu-cores", default="0-7")
    parser.add_argument("--no-telemetry", action="store_true")
    parser.add_argument("--warmup-ops", type=int, default=100_000)

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

    suite.run_waf()

    if not args.dry_run:
        suite.save(Path(args.output_dir) / "results.json")


if __name__ == "__main__":
    main()
