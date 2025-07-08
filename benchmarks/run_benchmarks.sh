#!/bin/bash
# Reproducible benchmark runner script

set -euo pipefail

# Configuration
ITERATIONS="${ITERATIONS:-5}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="benchmark_results"
LOG_FILE="${RESULTS_DIR}/benchmark_${TIMESTAMP}.log"

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Function to log
log() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "${LOG_FILE}"
}

# Start
log "Starting Lean benchmark suite"
log "Iterations: ${ITERATIONS}"
log "Results directory: ${RESULTS_DIR}"

# System info
log "System information:"
uname -a | tee -a "${LOG_FILE}"
lean --version 2>&1 | tee -a "${LOG_FILE}" || log "Lean not found"

# Create standard benchmarks
log "Creating standard benchmark files..."
python3 benchmarks/lean_benchmark_runner.py --create-standard

# Run benchmarks
log "Running benchmarks..."
BENCHMARK_FILE=$(python3 benchmarks/lean_benchmark_runner.py \
    --iterations "${ITERATIONS}" \
    2>&1 | tee -a "${LOG_FILE}" | \
    grep "Results saved to:" | \
    awk '{print $NF}')

if [ -z "${BENCHMARK_FILE}" ]; then
    log "ERROR: Failed to run benchmarks"
    exit 1
fi

log "Benchmark complete: ${BENCHMARK_FILE}"

# Analyze results
log "Analyzing results..."

# Generate CSV
CSV_FILE="${RESULTS_DIR}/benchmark_${TIMESTAMP}.csv"
python3 benchmarks/analyze_benchmarks.py "${BENCHMARK_FILE}" \
    --csv "${CSV_FILE}" 2>&1 | tee -a "${LOG_FILE}"

# Generate JSON summary
JSON_FILE="${RESULTS_DIR}/benchmark_${TIMESTAMP}_summary.json"
python3 benchmarks/analyze_benchmarks.py "${BENCHMARK_FILE}" \
    --json "${JSON_FILE}" 2>&1 | tee -a "${LOG_FILE}"

# Final summary
log "Benchmark suite complete!"
log "Raw data: ${BENCHMARK_FILE}"
log "CSV analysis: ${CSV_FILE}"
log "JSON summary: ${JSON_FILE}"
log "Full log: ${LOG_FILE}"

# Display quick summary
echo
echo "Quick Summary:"
python3 -c "
import json
with open('${JSON_FILE}', 'r') as f:
    data = json.load(f)
    for file, results in data['results'].items():
        print(f'{file}:')
        print(f'  Wall time: {results[\"statistics\"][\"wall_time\"][\"mean\"]:.3f}s (±{results[\"statistics\"][\"wall_time\"][\"stdev\"]:.3f}s)')
        print(f'  Memory: {results[\"statistics\"][\"memory_mb\"][\"mean\"]:.1f}MB')
"