# Simpulse Correctness Validator

The correctness validator ensures that Simpulse optimizations preserve the correctness of Lean proofs. It applies optimizations incrementally and validates that the code still compiles after each change.

## Features

- **Incremental Validation**: Applies optimizations one by one to identify which changes break compilation
- **Automatic Rollback**: Reverts changes that cause compilation failures
- **Batch Testing**: Validates multiple files in parallel
- **Safety Analysis**: Categorizes optimization rules as safe, unsafe, or conditional
- **Detailed Reporting**: Generates comprehensive reports on optimization success rates

## Usage

### Basic Usage

```python
from simpulse.validator.correctness import CorrectnessValidator

# Create validator
validator = CorrectnessValidator()

# Define optimizations
optimizations = [
    {
        'rule': 'add_zero',
        'location': 'line 10',
        'line': 10,
        'original': '@[simp] theorem add_zero',
        'replacement': '@[simp, priority := 500] theorem add_zero'
    }
]

# Validate a single file
result = validator.validate_file(lean_file_path, optimizations)

print(f"Success rate: {result.success_rate:.1%}")
print(f"Speedup: {result.speedup:.2f}x")
```

### Batch Validation

```python
# Prepare multiple files
files_and_optimizations = [
    (file1_path, optimizations1),
    (file2_path, optimizations2),
    # ...
]

# Run batch validation
report = validator.validate_batch(files_and_optimizations)

# Generate safety report
validation_results = [...]  # Collect from individual validations
safety_report = validator.generate_safety_report(validation_results)
```

### Integration with Optimizer

```python
from simpulse.optimizer import PriorityOptimizer

# Create optimizer with validation enabled
optimizer = PriorityOptimizer(validate_correctness=True)

# Analyze and optimize with safety checks
analysis_result = analyzer.analyze_project(project_path)
optimization_report = optimizer.optimize_with_safety_check(
    analysis_result,
    output_dir=Path("reports")
)
```

## CLI Usage

The validator is integrated into the Simpulse CLI:

```bash
# Optimize with correctness validation (default)
simpulse optimize /path/to/project

# Optimize without validation (faster but less safe)
simpulse optimize /path/to/project --no-validate

# Save safety report
simpulse optimize /path/to/project --safety-report safety_analysis.json
```

## How It Works

1. **Setup**: Creates a temporary workspace to avoid modifying original files
2. **Baseline**: Compiles the original file to establish baseline compilation time
3. **Incremental Application**: 
   - Applies each optimization individually
   - Runs `lake build` after each change
   - Records success/failure and compilation time
4. **Rollback**: If an optimization fails, reverts to the last working state
5. **Final State**: Keeps only the optimizations that preserve correctness

## Safety Categories

The validator categorizes optimization rules into three safety levels:

- **SAFE**: Never caused compilation failures
- **UNSAFE**: Always caused compilation failures
- **CONDITIONAL**: Sometimes safe, sometimes unsafe (context-dependent)

## Reports

The validator generates two types of reports:

### Batch Validation Report
- Total files tested
- Success rates per file
- Overall optimization statistics
- Failed optimization details

### Safety Report
- Rule safety categorization
- Safety scores per rule
- Recommendations for each rule

## Requirements

- Lean 4 installed and available in PATH
- `lake` build tool
- Valid Lean project with `lakefile.lean`

## Error Handling

The validator handles various error scenarios:

- Missing Lean installation
- Compilation timeouts
- Invalid file paths
- Malformed optimizations
- Project configuration issues

All errors are logged and included in the validation report.