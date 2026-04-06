#!/usr/bin/env bash

# Strict mode: Exit on error, pipe failure, or undefined variables
set -euo pipefail

# --- Configuration ---
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default paths (Adjust if your binary is elsewhere)
ROCKSDB_DIR="${ROCKSDB_DIR:-$HOME/Projects/rocksdb}"
DB_BENCH="$ROCKSDB_DIR/db_bench"
PYTHON_SUITE="$SCRIPT_DIR/benchmark_suite.py"
RESULTS_DIR="$SCRIPT_DIR/acs_results"

# Colors for logging
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_err() { echo -e "${RED}[ERROR]${NC} $1"; }

# --- Validation ---
check_env() {
  log_info "Checking environment..."

  if [[ ! -f "$DB_BENCH" ]]; then
    log_err "db_bench binary not found at: $DB_BENCH"
    echo "Please run 'make db_bench' or set ROCKSDB_DIR."
    exit 1
  fi

  if [[ ! -f "$PYTHON_SUITE" ]]; then
    log_err "Python suite not found at: $PYTHON_SUITE"
    exit 1
  fi

  if ! command -v python3 &>/dev/null; then
    log_err "python3 is not installed or not in PATH."
    exit 1
  fi

  log_ok "Environment ready."
}

# --- Execution Modes ---

run_fast() {
  log_info "Starting FAST (Dry Run) verification..."
  python3 "$PYTHON_SUITE" \
    --db-bench "$DB_BENCH" \
    --output-dir "$RESULTS_DIR/fast_run" \
    --repeat 1 \
    --fast
  log_ok "Fast verification complete."
}

run_full() {
  log_info "Starting FULL benchmark suite (3 repeats)..."
  # Ensure results dir is clean for full run
  mkdir -p "$RESULTS_DIR/full_run"

  python3 "$PYTHON_SUITE" \
    --db-bench "$DB_BENCH" \
    --output-dir "$RESULTS_DIR/full_run" \
    --repeat 3

  log_ok "Full benchmark complete. Results stored in $RESULTS_DIR/full_run"
}

# --- Main Logic ---

usage() {
  echo "Usage: $0 {fast|full|clean}"
  echo "  fast  : Run 1 iteration with low operation count to verify your code works."
  echo "  full  : Run the complete benchmark suite for your thesis."
  echo "  clean : Remove the results directory."
  exit 1
}

if [[ $# -eq 0 ]]; then
  usage
fi

check_env

case "$1" in
fast)
  run_fast
  ;;
full)
  run_full
  ;;
clean)
  log_info "Cleaning results..."
  rm -rf "$RESULTS_DIR"
  log_ok "Results directory removed."
  ;;
*)
  usage
  ;;
esac
