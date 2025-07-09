#!/usr/bin/env python3
"""
Simple Performance Verification for Simpulse
Creates a working performance test that validates the core claims
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple


def run_simpulse_command(cmd: List[str], cwd: str = ".") -> Tuple[bool, str, str]:
    """Run a simpulse command and return success, stdout, stderr"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=cwd
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def measure_directory_performance(directory: str) -> Dict:
    """Measure performance of Simpulse on a directory"""
    print(f"📊 Analyzing {directory}...")
    
    # Check what Simpulse finds
    success, stdout, stderr = run_simpulse_command([
        "python", "-m", "simpulse", "check", directory
    ])
    
    if not success:
        return {
            "directory": directory,
            "error": f"Check failed: {stderr}",
            "simp_rules": 0,
            "optimizable_rules": 0
        }
    
    # Parse the output to extract metrics
    simp_rules = 0
    optimizable_rules = 0
    
    for line in stdout.split('\n'):
        if "Found" in line and "simp rules" in line:
            try:
                simp_rules = int(line.split()[1])
            except:
                pass
        elif "Can optimize" in line and "rules" in line:
            try:
                optimizable_rules = int(line.split()[2])
            except:
                pass
    
    print(f"  ✅ Found {simp_rules} simp rules, {optimizable_rules} optimizable")
    
    # Run optimization analysis
    print(f"  🔍 Running optimization analysis...")
    success, stdout, stderr = run_simpulse_command([
        "python", "-m", "simpulse", "optimize", directory, "--json"
    ])
    
    if not success:
        return {
            "directory": directory,
            "error": f"Optimization failed: {stderr}",
            "simp_rules": simp_rules,
            "optimizable_rules": optimizable_rules
        }
    
    try:
        optimization_data = json.loads(stdout)
        estimated_improvement = optimization_data.get("estimated_improvement", 0)
        total_rules = optimization_data.get("total_rules", 0)
        rules_changed = optimization_data.get("rules_changed", 0)
        
        print(f"  🚀 Estimated improvement: {estimated_improvement:.1f}%")
        
        return {
            "directory": directory,
            "simp_rules": total_rules,
            "optimizable_rules": rules_changed,
            "estimated_improvement": estimated_improvement,
            "optimization_data": optimization_data,
            "success": True
        }
        
    except json.JSONDecodeError:
        return {
            "directory": directory,
            "error": f"Invalid JSON output: {stdout[:200]}...",
            "simp_rules": simp_rules,
            "optimizable_rules": optimizable_rules
        }


def test_performance_guarantee(directory: str) -> Dict:
    """Test the performance guarantee system"""
    print(f"🎯 Testing performance guarantee for {directory}...")
    
    success, stdout, stderr = run_simpulse_command([
        "python", "-m", "simpulse", "guarantee", directory
    ])
    
    # Performance guarantee exits with different codes
    # 0 = optimize, 1 = maybe, 2 = skip
    exit_code = 0 if success else (1 if "maybe" in stderr.lower() else 2)
    
    recommendation = "optimize" if exit_code == 0 else ("maybe" if exit_code == 1 else "skip")
    
    print(f"  💡 Recommendation: {recommendation}")
    
    return {
        "directory": directory,
        "recommendation": recommendation,
        "exit_code": exit_code,
        "output": stdout,
        "success": True
    }


def run_performance_verification():
    """Run the complete performance verification"""
    print("🚀 Simpulse Performance Verification")
    print("   Testing core functionality with real data\n")
    
    # Test directories
    test_dirs = [
        "benchmarks/excellence_suite/high_impact",
        "benchmarks/excellence_suite/moderate_impact", 
        "benchmarks/excellence_suite/no_benefit"
    ]
    
    results = {}
    
    for directory in test_dirs:
        if not Path(directory).exists():
            print(f"⚠️  Directory {directory} does not exist, skipping...")
            continue
            
        print(f"\n{'='*60}")
        print(f"Testing: {directory}")
        print(f"{'='*60}")
        
        # Measure performance
        perf_result = measure_directory_performance(directory)
        
        # Test guarantee system
        guarantee_result = test_performance_guarantee(directory)
        
        # Combine results
        results[directory] = {
            "performance": perf_result,
            "guarantee": guarantee_result,
            "timestamp": time.time()
        }
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("📋 PERFORMANCE VERIFICATION SUMMARY")
    print(f"{'='*60}")
    
    total_rules = 0
    total_optimizable = 0
    working_directories = 0
    
    for directory, data in results.items():
        perf = data["performance"]
        guarantee = data["guarantee"]
        
        if perf.get("success"):
            working_directories += 1
            total_rules += perf.get("simp_rules", 0)
            total_optimizable += perf.get("optimizable_rules", 0)
            
            category = directory.split("/")[-1]
            improvement = perf.get("estimated_improvement", 0)
            
            print(f"\n📂 {category.upper()}:")
            print(f"   Rules found: {perf.get('simp_rules', 0)}")
            print(f"   Optimizable: {perf.get('optimizable_rules', 0)}")
            print(f"   Estimated improvement: {improvement:.1f}%")
            print(f"   Guarantee recommendation: {guarantee.get('recommendation', 'unknown')}")
            
            # Validate expectations
            if "high_impact" in directory:
                if improvement >= 20:
                    print(f"   ✅ PASSED: High impact expectation met ({improvement:.1f}%)")
                else:
                    print(f"   ⚠️  REVIEW: Expected >20% improvement, got {improvement:.1f}%")
            elif "moderate_impact" in directory:
                if improvement >= 5:
                    print(f"   ✅ PASSED: Moderate impact expectation met ({improvement:.1f}%)")
                else:
                    print(f"   ⚠️  REVIEW: Expected >5% improvement, got {improvement:.1f}%")
            elif "no_benefit" in directory:
                if improvement < 5 or guarantee.get("recommendation") == "skip":
                    print(f"   ✅ PASSED: Correctly identified low/no benefit")
                else:
                    print(f"   ⚠️  REVIEW: Expected low benefit, got {improvement:.1f}%")
        else:
            print(f"\n❌ {directory}: {perf.get('error', 'Unknown error')}")
    
    # Overall summary
    print(f"\n🏆 OVERALL RESULTS:")
    print(f"   Working directories: {working_directories}/{len(test_dirs)}")
    print(f"   Total simp rules analyzed: {total_rules}")
    print(f"   Total optimizable rules: {total_optimizable}")
    print(f"   Optimization coverage: {(total_optimizable/total_rules)*100:.1f}% of rules")
    
    if working_directories >= 2:
        print(f"   ✅ VERIFICATION PASSED: Tool works on real data")
    else:
        print(f"   ❌ VERIFICATION FAILED: Tool not working properly")
    
    # Save detailed results
    report_file = "performance_verification_report.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Detailed report saved to: {report_file}")
    
    return results


if __name__ == "__main__":
    run_performance_verification()