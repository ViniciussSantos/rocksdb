#!/usr/bin/env bash

# Strict mode: Exit on error, pipe failure, or undefined variables
set -euo pipefail

# --- Configuration ---
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default paths
ROCKSDB_DIR="${ROCKSDB_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DB_BENCH="$ROCKSDB_DIR/db_bench"
PYTHON_SUITE="$SCRIPT_DIR/benchmark_suite.py"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/acs_results}"

# Default benchmark parameters
REPEAT="${REPEAT:-3}"
CPU_CORES="${CPU_CORES:-0-7}"
WARMUP_OPS="${WARMUP_OPS:-100000}"
EXPERIMENTS="${EXPERIMENTS:-all}"

# Colors for logging
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_err() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# --- Validation ---
check_env() {
  log_info "Checking environment..."

  log_info "Detected paths:"
  log_info "  RocksDB dir: $ROCKSDB_DIR"
  log_info "  db_bench: $DB_BENCH"
  log_info "  Python suite: $PYTHON_SUITE"
  log_info "  Results dir: $RESULTS_DIR"
  echo ""

  if [[ ! -f "$DB_BENCH" ]]; then
    log_err "db_bench binary not found at: $DB_BENCH"
    echo "Please run 'make db_bench' in RocksDB directory or set ROCKSDB_DIR."
    exit 1
  fi

  if [[ ! -f "$PYTHON_SUITE" ]]; then
    log_err "Python suite not found at: $PYTHON_SUITE"
    echo "Make sure benchmark_suite.py is in the same directory as this script."
    exit 1
  fi

  if ! command -v python3 &>/dev/null; then
    log_err "python3 is not installed or not in PATH."
    exit 1
  fi

  # Check for required Python packages
  if ! python3 -c "import numpy" &>/dev/null; then
    log_warn "numpy not found - some features may not work"
    echo "Install with: pip3 install numpy"
  fi

  # Check for optional telemetry tools
  if ! command -v pidstat &>/dev/null; then
    log_warn "pidstat not found - CPU telemetry will be disabled"
    echo "Install with: apt-get install sysstat (Ubuntu/Debian)"
  fi

  if ! command -v iostat &>/dev/null; then
    log_warn "iostat not found - I/O telemetry will be disabled"
    echo "Install with: apt-get install sysstat (Ubuntu/Debian)"
  fi

  log_ok "Environment ready."
  echo ""
}

# --- Execution Modes ---

run_validate() {
  log_info "Running validation test (quick sanity check)..."

  python3 "$PYTHON_SUITE" \
    --db-bench "$DB_BENCH" \
    --output-dir "$RESULTS_DIR/validation" \
    --repeat 1 \
    --experiments waf \
    --cpu-cores "$CPU_CORES" \
    --warmup-ops 10000 \
    --dry-run

  log_ok "Validation complete - dry run successful"
  log_info "To run actual test, use: $0 fast"
}

run_fast() {
  log_info "Starting FAST test (1 iteration, reduced ops)..."

  python3 "$PYTHON_SUITE" \
    --db-bench "$DB_BENCH" \
    --output-dir "$RESULTS_DIR/fast_run" \
    --repeat 1 \
    --experiments "${EXPERIMENTS}" \
    --cpu-cores "$CPU_CORES" \
    --warmup-ops "$WARMUP_OPS"

  log_ok "Fast test complete. Results in $RESULTS_DIR/fast_run"
}

run_full() {
  log_info "Starting FULL benchmark suite..."
  log_info "Configuration:"
  log_info "  Repeats: $REPEAT"
  log_info "  CPU cores: $CPU_CORES"
  log_info "  Experiments: $EXPERIMENTS"
  log_info "  Warmup ops: $WARMUP_OPS"
  echo ""

  read -p "This will take several hours. Continue? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_info "Cancelled by user"
    exit 0
  fi

  mkdir -p "$RESULTS_DIR/full_run"

  python3 "$PYTHON_SUITE" \
    --db-bench "$DB_BENCH" \
    --output-dir "$RESULTS_DIR/full_run" \
    --repeat "$REPEAT" \
    --experiments $EXPERIMENTS \
    --cpu-cores "$CPU_CORES" \
    --warmup-ops "$WARMUP_OPS"

  log_ok "Full benchmark complete. Results in $RESULTS_DIR/full_run"
}

run_experiment() {
  local exp_name=$1
  log_info "Running $exp_name experiment..."

  python3 "$PYTHON_SUITE" \
    --db-bench "$DB_BENCH" \
    --output-dir "$RESULTS_DIR/${exp_name}_run" \
    --repeat "$REPEAT" \
    --experiments "$exp_name" \
    --cpu-cores "$CPU_CORES" \
    --warmup-ops "$WARMUP_OPS"

  log_ok "$exp_name experiment complete. Results in $RESULTS_DIR/${exp_name}_run"
}

# --- Main Logic ---

usage() {
  cat <<EOF
Usage: $0 {validate|fast|full|experiment|clean} [OPTIONS]

MODES:
  validate    Dry run to verify setup (no actual execution)
  fast        Run 1 iteration with reduced ops (quick test)
  full        Run complete benchmark suite (for thesis)
  experiment  Run a specific experiment
  clean       Remove results directory

EXAMPLES:
  # Quick validation
  $0 validate
  
  # Fast test with specific experiments
  EXPERIMENTS="waf latency" $0 fast
  
  # Full suite with 3 repeats
  REPEAT=3 $0 full
  
  # Run only WAF experiment
  $0 experiment waf
  
  # Full suite on specific cores
  CPU_CORES="0-15" REPEAT=3 $0 full

ENVIRONMENT VARIABLES:
  ROCKSDB_DIR      RocksDB root directory (default: auto-detected)
  RESULTS_DIR      Output directory (default: $SCRIPT_DIR/acs_results)
  REPEAT           Number of repeats (default: 3)
  CPU_CORES        CPU cores to use (default: 0-7)
  WARMUP_OPS       Warmup operations (default: 100000)
  EXPERIMENTS      Which experiments: waf, latency, scaling, mixed, burst, all (default: all)

AVAILABLE EXPERIMENTS:
  waf       Write Amplification Factor comparison
  latency   Latency under heavy load
  scaling   Thread scaling efficiency
  mixed     Mixed read/write workload
  burst     Bursty write pattern
  all       Run all experiments

EOF
  exit 1
}

if [[ $# -eq 0 ]]; then
  usage
fi

check_env

case "$1" in
validate)
  run_validate
  ;;
fast)
  run_fast
  ;;
full)
  run_full
  ;;
experiment)
  if [[ $# -lt 2 ]]; then
    log_err "experiment mode requires an experiment name"
    echo "Available: waf, latency, scaling, mixed, burst"
    exit 1
  fi

  # Validate experiment name
  case "$2" in
  waf | latency | scaling | mixed | burst)
    EXPERIMENTS="$2"
    run_experiment "$2"
    ;;
  *)
    log_err "Unknown experiment: $2"
    echo "Available: waf, latency, scaling, mixed, burst"
    exit 1
    ;;
  esac
  ;;
clean)
  log_info "Cleaning results..."
  rm -rf "$RESULTS_DIR"
  log_ok "Results directory removed: $RESULTS_DIR"
  ;;
*)
  usage
  ;;
esac
