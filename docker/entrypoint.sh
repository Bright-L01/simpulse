#!/bin/bash
set -e

# Simpulse Docker Entrypoint Script
# This script handles GitHub Action execution and CLI mode

echo "🧬 Simpulse Optimizer Container"
echo "==============================="

# Parse arguments from GitHub Action
MODULES="${1:-auto}"
TIME_BUDGET="${2:-7200}"
TARGET_IMPROVEMENT="${3:-15}"
POPULATION_SIZE="${4:-30}"
MAX_GENERATIONS="${5:-50}"
CREATE_PR="${6:-true}"
PR_BRANCH="${7:-}"
BASE_BRANCH="${8:-main}"
CLAUDE_BACKEND="${9:-claude_code}"
WORKING_DIR="${10:-.}"
CACHE_ENABLED="${11:-true}"
PROGRESS_COMMENTS="${12:-true}"
REPORT_FORMAT="${13:-both}"
PARALLEL_WORKERS="${14:-4}"
ENABLE_TELEMETRY="${15:-true}"
DRY_RUN="${16:-false}"

# Setup environment
export PYTHONPATH="/app/src:$PYTHONPATH"

# Validate environment
echo "🔍 Validating environment..."

# Check if running in GitHub Actions
if [ -n "$GITHUB_ACTIONS" ]; then
    echo "✓ Running in GitHub Actions"
    echo "  Repository: $GITHUB_REPOSITORY"
    echo "  SHA: $GITHUB_SHA"
    echo "  Actor: $GITHUB_ACTOR"
else
    echo "ℹ️ Running in standalone mode"
fi

# Check Lean installation
if command -v lean >/dev/null 2>&1; then
    LEAN_VERSION=$(lean --version)
    echo "✓ Lean installed: $LEAN_VERSION"
else
    echo "❌ Lean not found"
    exit 1
fi

# Check Lake installation
if command -v lake >/dev/null 2>&1; then
    LAKE_VERSION=$(lake --version)
    echo "✓ Lake installed: $LAKE_VERSION"
else
    echo "❌ Lake not found"
    exit 1
fi

# Check Python dependencies
python3 -c "import simpulse" 2>/dev/null && echo "✓ Simpulse Python package available" || {
    echo "❌ Simpulse package not available"
    exit 1
}

# Check Claude Code CLI (optional)
if command -v claude >/dev/null 2>&1; then
    CLAUDE_VERSION=$(claude --version 2>/dev/null || echo "unknown")
    echo "✓ Claude Code CLI available: $CLAUDE_VERSION"
    export CLAUDE_AVAILABLE=true
else
    echo "⚠️ Claude Code CLI not available (will use API fallback)"
    export CLAUDE_AVAILABLE=false
fi

# Change to working directory
cd "$WORKING_DIR"

# Validate Lean project
if [ -f "lakefile.lean" ] || [ -f "lakefile.toml" ]; then
    echo "✓ Lean project detected"
else
    echo "⚠️ No lakefile found - may not be a Lean project"
fi

# Setup cache directory
if [ "$CACHE_ENABLED" = "true" ]; then
    mkdir -p "$SIMPULSE_CACHE_DIR"
    echo "✓ Cache enabled: $SIMPULSE_CACHE_DIR"
fi

# Setup logging
mkdir -p "$SIMPULSE_LOG_DIR"
export SIMPULSE_LOG_LEVEL="${SIMPULSE_LOG_LEVEL:-INFO}"

echo ""
echo "🚀 Starting optimization..."
echo "  Modules: $MODULES"
echo "  Time budget: ${TIME_BUDGET}s"
echo "  Target improvement: ${TARGET_IMPROVEMENT}%"
echo "  Population size: $POPULATION_SIZE"
echo "  Max generations: $MAX_GENERATIONS"
echo "  Parallel workers: $PARALLEL_WORKERS"
echo "  Create PR: $CREATE_PR"
echo "  Dry run: $DRY_RUN"
echo ""

# Build optimization command
CMD_ARGS=(
    "python3" "-m" "simpulse.cli_v2"
    "optimize"
    "--modules" "$MODULES"
    "--time-budget" "$TIME_BUDGET"
    "--target-improvement" "$TARGET_IMPROVEMENT"
    "--population-size" "$POPULATION_SIZE"
    "--max-generations" "$MAX_GENERATIONS"
    "--parallel-workers" "$PARALLEL_WORKERS"
    "--claude-backend" "$CLAUDE_BACKEND"
    "--output-dir" "$SIMPULSE_REPORTS_DIR"
)

# Add conditional flags
if [ "$CREATE_PR" = "true" ]; then
    CMD_ARGS+=("--create-pr")
    if [ -n "$PR_BRANCH" ]; then
        CMD_ARGS+=("--pr-branch" "$PR_BRANCH")
    fi
    CMD_ARGS+=("--base-branch" "$BASE_BRANCH")
fi

if [ "$CACHE_ENABLED" = "true" ]; then
    CMD_ARGS+=("--cache-dir" "$SIMPULSE_CACHE_DIR")
fi

if [ "$PROGRESS_COMMENTS" = "true" ]; then
    CMD_ARGS+=("--progress-comments")
fi

if [ "$ENABLE_TELEMETRY" = "true" ]; then
    CMD_ARGS+=("--enable-telemetry")
fi

if [ "$DRY_RUN" = "true" ]; then
    CMD_ARGS+=("--dry-run")
fi

CMD_ARGS+=("--report-format" "$REPORT_FORMAT")

# Execute optimization
echo "🧬 Executing: ${CMD_ARGS[*]}"
echo ""

# Run with proper error handling
if "${CMD_ARGS[@]}"; then
    OPTIMIZATION_SUCCESS=true
    echo ""
    echo "✅ Optimization completed successfully!"
else
    OPTIMIZATION_SUCCESS=false
    RETURN_CODE=$?
    echo ""
    echo "❌ Optimization failed with exit code: $RETURN_CODE"
fi

# Generate GitHub Actions outputs
if [ -n "$GITHUB_ACTIONS" ]; then
    echo "📝 Setting GitHub Actions outputs..."
    
    # Read results from output files if available
    RESULTS_FILE="$SIMPULSE_REPORTS_DIR/results.json"
    
    if [ -f "$RESULTS_FILE" ]; then
        # Extract values from results
        IMPROVEMENT=$(python3 -c "
import json, sys
try:
    with open('$RESULTS_FILE') as f:
        data = json.load(f)
    print(data.get('improvement_percent', 0))
except:
    print(0)
")
        
        GENERATIONS=$(python3 -c "
import json, sys
try:
    with open('$RESULTS_FILE') as f:
        data = json.load(f)
    print(data.get('total_generations', 0))
except:
    print(0)
")
        
        EXEC_TIME=$(python3 -c "
import json, sys
try:
    with open('$RESULTS_FILE') as f:
        data = json.load(f)
    print(data.get('execution_time', 0))
except:
    print(0)
")
        
        MUTATIONS=$(python3 -c "
import json, sys
try:
    with open('$RESULTS_FILE') as f:
        data = json.load(f)
    best = data.get('best_candidate', {})
    print(len(best.get('mutations', [])))
except:
    print(0)
")
        
        PR_URL=$(python3 -c "
import json, sys
try:
    with open('$RESULTS_FILE') as f:
        data = json.load(f)
    print(data.get('pr_url', ''))
except:
    print('')
")
        
    else
        echo "⚠️ Results file not found, using default values"
        IMPROVEMENT=0
        GENERATIONS=0
        EXEC_TIME=0
        MUTATIONS=0
        PR_URL=""
    fi
    
    # Set outputs for GitHub Actions
    echo "improvement-percent=$IMPROVEMENT" >> $GITHUB_OUTPUT
    echo "total-generations=$GENERATIONS" >> $GITHUB_OUTPUT
    echo "execution-time=$EXEC_TIME" >> $GITHUB_OUTPUT
    echo "mutations-applied=$MUTATIONS" >> $GITHUB_OUTPUT
    echo "pr-url=$PR_URL" >> $GITHUB_OUTPUT
    echo "success=$OPTIMIZATION_SUCCESS" >> $GITHUB_OUTPUT
    
    # Find report file
    REPORT_FILE=""
    if [ -f "$SIMPULSE_REPORTS_DIR/optimization_report.html" ]; then
        REPORT_FILE="$SIMPULSE_REPORTS_DIR/optimization_report.html"
    elif [ -f "$SIMPULSE_REPORTS_DIR/optimization_report.md" ]; then
        REPORT_FILE="$SIMPULSE_REPORTS_DIR/optimization_report.md"
    fi
    
    echo "report-path=$REPORT_FILE" >> $GITHUB_OUTPUT
    
    echo "✓ GitHub Actions outputs set"
fi

# Cleanup
echo ""
echo "🧹 Cleaning up..."

# Archive logs and reports if successful
if [ "$OPTIMIZATION_SUCCESS" = true ]; then
    echo "📦 Archiving results..."
    # In a real scenario, might upload artifacts or send notifications
fi

# Summary
echo ""
echo "📊 Summary:"
echo "  Success: $OPTIMIZATION_SUCCESS"
if [ -n "$IMPROVEMENT" ] && [ "$IMPROVEMENT" != "0" ]; then
    echo "  Improvement: ${IMPROVEMENT}%"
    echo "  Generations: $GENERATIONS"
    echo "  Execution time: ${EXEC_TIME}s"
    echo "  Mutations applied: $MUTATIONS"
fi

if [ "$OPTIMIZATION_SUCCESS" = true ]; then
    echo ""
    echo "🎉 Simpulse optimization completed successfully!"
    exit 0
else
    echo ""
    echo "💥 Simpulse optimization failed"
    exit 1
fi