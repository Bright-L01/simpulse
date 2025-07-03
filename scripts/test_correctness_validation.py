#!/usr/bin/env python3
"""
Test the correctness validation system on multiple Lean files.

This script demonstrates the batch testing capability of the correctness validator.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simpulse.analyzer import LeanAnalyzer
from src.simpulse.optimizer import PriorityOptimizer
from src.simpulse.validator.correctness import CorrectnessValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_lean_files(directory: Path, max_files: int = 10) -> list[Path]:
    """Find Lean files in a directory for testing."""
    lean_files = []

    for file_path in directory.rglob("*.lean"):
        # Skip build directories and generated files
        if any(part in file_path.parts for part in [".lake", "build", "_target"]):
            continue

        # Skip very large files
        if file_path.stat().st_size > 100000:  # 100KB
            continue

        lean_files.append(file_path)

        if len(lean_files) >= max_files:
            break

    return lean_files


def main():
    """Run batch correctness validation test."""
    # Test locations
    test_dirs = [
        Path(__file__).parent.parent / "lean4" / "Benchmark",
        Path(__file__).parent.parent / "benchmarks" / "lean_test_files",
        Path(__file__).parent.parent / "lean4",
    ]

    # Collect test files
    test_files = []
    for test_dir in test_dirs:
        if test_dir.exists():
            files = find_lean_files(test_dir, max_files=5)
            test_files.extend(files)
            logger.info(f"Found {len(files)} test files in {test_dir}")

    if not test_files:
        logger.error("No test files found!")
        return

    logger.info(f"Testing correctness validation on {len(test_files)} files")

    # Initialize components
    analyzer = LeanAnalyzer()
    optimizer = PriorityOptimizer(validate_correctness=True)
    validator = CorrectnessValidator()

    # Prepare files and optimizations for batch validation
    files_and_optimizations = []

    for file_path in test_files:
        logger.info(f"\nAnalyzing {file_path}...")

        try:
            # Analyze the file
            analysis = analyzer.analyze_file(file_path)

            if not analysis or not analysis.get("rules"):
                logger.info(f"No simp rules found in {file_path}")
                continue

            # Generate optimization suggestions
            suggestions = optimizer.optimize_project(analysis)

            if not suggestions:
                logger.info(f"No optimization suggestions for {file_path}")
                continue

            # Convert suggestions to optimization format
            optimizations = []
            for suggestion in suggestions[:5]:  # Limit to 5 optimizations per file
                optimizations.append(
                    {
                        "rule": suggestion.rule_name,
                        "location": f"theorem {suggestion.rule_name}",
                        "line": 1,  # Would need proper line extraction
                        "original": f"@[simp] theorem {suggestion.rule_name}",
                        "replacement": f"@[simp, priority := {suggestion.suggested_priority}] theorem {suggestion.rule_name}",
                        "priority": suggestion.suggested_priority,
                    }
                )

            files_and_optimizations.append((file_path, optimizations))
            logger.info(f"Generated {len(optimizations)} optimizations for {file_path}")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue

    if not files_and_optimizations:
        logger.error("No files with optimizations to test!")
        return

    # Run batch validation
    logger.info(f"\nRunning batch validation on {len(files_and_optimizations)} files...")

    batch_report = validator.validate_batch(files_and_optimizations)

    # Display results
    print("\n" + "=" * 60)
    print("CORRECTNESS VALIDATION REPORT")
    print("=" * 60)
    print(f"Timestamp: {batch_report['timestamp']}")
    print(f"Total files tested: {batch_report['total_files_tested']}")
    print(f"Files successfully optimized: {batch_report['files_successfully_optimized']}")
    print(f"Files that preserved correctness: {batch_report['files_preserved_correctness']}")
    print(f"Total optimizations attempted: {batch_report['total_optimizations_attempted']}")
    print(f"Successful optimizations: {batch_report['successful_optimizations']}")
    print(f"Overall success rate: {batch_report['overall_success_rate']:.1%}")
    print(f"Average speedup: {batch_report['average_speedup']:.2f}x")

    print("\n" + "-" * 60)
    print("FILE RESULTS:")
    print("-" * 60)

    for file_result in batch_report["file_results"]:
        print(f"\nFile: {Path(file_result['file']).name}")
        print(f"  Success rate: {file_result['success_rate']:.1%}")
        print(f"  Speedup: {file_result['speedup']:.2f}x")
        print(f"  Successful: {file_result['successful_optimizations']}")
        print(f"  Failed: {file_result['failed_optimizations']}")
        if file_result["error"]:
            print(f"  Error: {file_result['error']}")

    # Show failed optimizations
    if batch_report["failed_optimizations"]:
        print("\n" + "-" * 60)
        print("FAILED OPTIMIZATIONS:")
        print("-" * 60)

        for failed in batch_report["failed_optimizations"][:10]:  # Show first 10
            print(f"\nFile: {Path(failed['file']).name}")
            print(f"  Rule: {failed['optimization']}")
            print(f"  Location: {failed['location']}")
            print(f"  Error: {failed['error'][:100]}...")  # First 100 chars

    # Generate safety report
    logger.info("\nGenerating safety report...")

    # Collect all validation results
    validation_results = []
    for file_path, _ in files_and_optimizations:
        # Find corresponding result in batch report
        for file_result in batch_report["file_results"]:
            if file_result["file"] == str(file_path):
                # Mock a ValidationResult for safety analysis
                # In real usage, we'd collect these during validation
                from src.simpulse.validator.correctness import ValidationResult

                result = ValidationResult(
                    file_path=str(file_path),
                    original_compilation_time=1.0,
                    optimized_compilation_time=0.8,
                    total_optimizations=file_result["successful_optimizations"]
                    + file_result["failed_optimizations"],
                    successful_optimizations=file_result["successful_optimizations"],
                    failed_optimizations=file_result["failed_optimizations"],
                )
                validation_results.append(result)
                break

    if validation_results:
        safety_report = validator.generate_safety_report(validation_results)

        print("\n" + "=" * 60)
        print("SAFETY REPORT")
        print("=" * 60)
        print(f"Total rules tested: {safety_report['total_rules_tested']}")
        print(f"Safe rules: {len(safety_report['safe_rules'])}")
        print(f"Unsafe rules: {len(safety_report['unsafe_rules'])}")
        print(f"Conditional rules: {len(safety_report['conditional_rules'])}")

        # Save reports
        output_dir = Path("validation_reports")
        output_dir.mkdir(exist_ok=True)

        with open(
            output_dir / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w"
        ) as f:
            json.dump(batch_report, f, indent=2)

        with open(
            output_dir / f"safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w"
        ) as f:
            json.dump(safety_report, f, indent=2)

        print(f"\nReports saved to {output_dir}/")


if __name__ == "__main__":
    main()
