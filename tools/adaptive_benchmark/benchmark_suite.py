#!/usr/bin/env python3

import subprocess
import json
import time
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import argparse
import re


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
    def __init__(self, db_bench_path: str, output_dir: str, dry_run=False):
        self.db_bench_path = db_bench_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run

    def run_benchmark(self, config: BenchmarkConfig, benchmark_type="fillrandom"):

        db_path = self.output_dir / f"db_{config.name}"
        if db_path.exists():
            subprocess.run(["rm", "-rf", str(db_path)])
        db_path.mkdir(parents=True)

        cmd = [self.db_bench_path]
        cmd.append(f"--benchmarks={benchmark_type}")
        cmd.extend(config.to_db_bench_args(str(db_path)))

        print("\nCMD:", " ".join(cmd))

        if self.dry_run:
            return None

        output_file = self.output_dir / f"{config.name}.txt"

        start = time.time()
        with open(output_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        duration = time.time() - start

        metrics = self._parse_output(output_file)
        metrics["duration_sec"] = duration

        return {"config": config.name, "metrics": metrics}

    def _parse_output(self, file: Path) -> Dict:
        text = file.read_text()

        def extract(pattern, default=0.0):
            m = re.search(pattern, text)
            return float(m.group(1)) if m else default

        return {
            "ops_per_sec": extract(r"(\d+\.\d+)\s+ops/sec"),
            "p50_latency_us": extract(r"P50.*?:\s+(\d+)"),
            "p99_latency_us": extract(r"P99.*?:\s+(\d+)"),
            "p999_latency_us": extract(r"P99\.9.*?:\s+(\d+)"),
            "write_amplification": extract(r"write amplification.*?(\d+\.\d+)"),
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
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-bench", required=True)
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast", action="store_true")

    args = parser.parse_args()

    runner = BenchmarkRunner(args.db_bench, args.output_dir, args.dry_run)
    suite = ExperimentSuite(runner, args.repeat)

    suite.run_waf()

    if not args.dry_run:
        suite.save(Path(args.output_dir) / "results.json")


if __name__ == "__main__":
    main()
