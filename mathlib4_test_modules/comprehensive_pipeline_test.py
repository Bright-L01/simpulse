#!/usr/bin/env python3
"""
Comprehensive Simpulse pipeline test on real mathlib4 modules.
Tests extraction, analysis, optimization, and validation on production Lean 4 code.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simpulse.analyzer import LeanAnalyzer
from simpulse.evolution.rule_extractor import RuleExtractor
from simpulse.optimization.pattern_analyzer import PatternAnalyzer
from simpulse.optimization.smart_optimizer import SmartPatternOptimizer


class MathlibPipelineTest:
    """Complete pipeline test for real mathlib4 modules."""

    def __init__(self):
        self.analyzer = LeanAnalyzer()
        self.rule_extractor = RuleExtractor()
        self.pattern_analyzer = PatternAnalyzer()
        self.smart_optimizer = SmartPatternOptimizer()

    def test_single_module(self, module_path: Path, module_name: str):
        """Run complete pipeline on a single module."""
        print(f"\n{'='*60}")
        print(f"🔬 Testing: {module_name}")
        print(f"📁 File: {module_path.name}")
        print(f"{'='*60}")

        results = {
            "module_name": module_name,
            "file_path": str(module_path),
            "file_name": module_path.name,
            "timestamp": datetime.now().isoformat(),
            "stages": {},
        }

        # Get basic file statistics
        try:
            with open(module_path, encoding="utf-8") as f:
                content = f.read()

            basic_stats = {
                "total_lines": len(content.splitlines()),
                "file_size_bytes": len(content.encode("utf-8")),
                "theorem_count": content.count("theorem ") + content.count("lemma "),
                "simp_annotations": content.count("@[simp]"),
                "characters": len(content),
            }
            results["basic_stats"] = basic_stats

            print(f"📊 Basic Stats:")
            print(f"   Lines: {basic_stats['total_lines']:,}")
            print(f"   Theorems/Lemmas: {basic_stats['theorem_count']:,}")
            print(f"   @[simp] annotations: {basic_stats['simp_annotations']:,}")
            print(f"   File size: {basic_stats['file_size_bytes']:,} bytes")

        except Exception as e:
            results["stages"]["basic_analysis"] = {"status": "failed", "error": str(e)}
            return results

        # Stage 1: Rule Extraction with Enhanced Extractor
        print("\n🔍 Stage 1: Enhanced Rule Extraction")
        try:
            start_time = time.time()

            extraction_result = self.rule_extractor.extract_rules_from_file(module_path)

            stage_result = {
                "status": "success",
                "duration_seconds": time.time() - start_time,
                "rules_extracted": len(extraction_result.rules),
                "module_name": extraction_result.module_name,
                "has_imports": len(extraction_result.imports) > 0,
                "imports_count": len(extraction_result.imports),
            }

            # Analyze rule priorities and types
            priority_counts = {}
            direction_counts = {"forward": 0, "backward": 0}

            for rule in extraction_result.rules:
                # Count by priority
                if isinstance(rule.priority, int):
                    priority_key = f"numeric_{rule.priority}"
                else:
                    priority_key = rule.priority.value
                priority_counts[priority_key] = priority_counts.get(priority_key, 0) + 1

                # Count by direction
                direction_counts[rule.direction.value] += 1

            stage_result["priority_distribution"] = priority_counts
            stage_result["direction_distribution"] = direction_counts

            # Show sample rules
            sample_rules = []
            for i, rule in enumerate(extraction_result.rules[:5]):
                sample_rules.append(
                    {
                        "name": rule.name,
                        "line": rule.location.line if rule.location else "unknown",
                        "priority": (
                            rule.priority.value
                            if hasattr(rule.priority, "value")
                            else str(rule.priority)
                        ),
                        "direction": rule.direction.value,
                    }
                )
            stage_result["sample_rules"] = sample_rules

            results["stages"]["rule_extraction"] = stage_result

            print(f"✅ Extracted {len(extraction_result.rules)} simp rules")
            print(f"   Priority distribution: {priority_counts}")
            print(f"   Direction distribution: {direction_counts}")

        except Exception as e:
            results["stages"]["rule_extraction"] = {
                "status": "failed",
                "error": str(e),
                "duration_seconds": time.time() - start_time if "start_time" in locals() else 0,
            }
            print(f"❌ Rule extraction failed: {e}")

        # Stage 2: Traditional Analyzer (for comparison)
        print("\n🔍 Stage 2: Traditional Analyzer")
        try:
            start_time = time.time()

            analyzer_result = self.analyzer.analyze_file(module_path)

            stage_result = {
                "status": "success",
                "duration_seconds": time.time() - start_time,
                "rules_found": len(analyzer_result.simp_rules),
                "syntax_valid": analyzer_result.syntax_valid,
                "total_lines": analyzer_result.total_lines,
            }

            results["stages"]["traditional_analyzer"] = stage_result

            print(f"✅ Traditional analyzer found {len(analyzer_result.simp_rules)} rules")
            print(f"   Syntax validation: {'✅' if analyzer_result.syntax_valid else '❌'}")

        except Exception as e:
            results["stages"]["traditional_analyzer"] = {
                "status": "failed",
                "error": str(e),
                "duration_seconds": time.time() - start_time if "start_time" in locals() else 0,
            }
            print(f"❌ Traditional analyzer failed: {e}")

        # Stage 3: Pattern Analysis
        print("\n🧠 Stage 3: Pattern Analysis")
        try:
            start_time = time.time()

            # Create temporary directory for pattern analysis
            temp_dir = module_path.parent / f"temp_pattern_{module_path.stem}"
            temp_dir.mkdir(exist_ok=True)

            import shutil

            temp_file = temp_dir / module_path.name
            shutil.copy2(module_path, temp_file)

            # Run pattern analysis on the temporary directory
            pattern_result = self.pattern_analyzer.analyze_patterns(temp_dir)

            # Clean up
            shutil.rmtree(temp_dir)

            stage_result = {
                "status": "success",
                "duration_seconds": time.time() - start_time,
                "patterns_found": len(pattern_result.get("patterns", [])),
                "context_patterns": len(pattern_result.get("context_patterns", [])),
                "rule_patterns": len(pattern_result.get("rule_patterns", [])),
            }

            results["stages"]["pattern_analysis"] = stage_result

            print(f"✅ Pattern analysis found {stage_result['patterns_found']} patterns")
            print(f"   Context patterns: {stage_result['context_patterns']}")
            print(f"   Rule patterns: {stage_result['rule_patterns']}")

        except Exception as e:
            results["stages"]["pattern_analysis"] = {
                "status": "failed",
                "error": str(e),
                "duration_seconds": time.time() - start_time if "start_time" in locals() else 0,
            }
            print(f"❌ Pattern analysis failed: {e}")

        # Stage 4: Smart Optimization
        print("\n⚡ Stage 4: Smart Optimization")
        try:
            start_time = time.time()

            # Create temporary directory for optimization
            temp_dir = module_path.parent / f"temp_optimize_{module_path.stem}"
            temp_dir.mkdir(exist_ok=True)

            import shutil

            temp_file = temp_dir / module_path.name
            shutil.copy2(module_path, temp_file)

            # Run smart optimization
            optimization_result = self.smart_optimizer.analyze(temp_dir)

            # Clean up
            shutil.rmtree(temp_dir)

            stage_result = {
                "status": "success",
                "duration_seconds": time.time() - start_time,
                "optimization_suggestions": len(optimization_result.get("optimizations", [])),
                "insights_generated": len(optimization_result.get("insights", [])),
                "potential_improvements": len(optimization_result.get("improvements", [])),
            }

            results["stages"]["smart_optimization"] = stage_result

            print(
                f"✅ Smart optimization generated {stage_result['optimization_suggestions']} suggestions"
            )
            print(f"   Insights: {stage_result['insights_generated']}")
            print(f"   Potential improvements: {stage_result['potential_improvements']}")

        except Exception as e:
            results["stages"]["smart_optimization"] = {
                "status": "failed",
                "error": str(e),
                "duration_seconds": time.time() - start_time if "start_time" in locals() else 0,
            }
            print(f"❌ Smart optimization failed: {e}")

        # Calculate overall metrics
        successful_stages = sum(
            1 for stage in results["stages"].values() if stage.get("status") == "success"
        )
        total_stages = len(results["stages"])
        total_duration = sum(
            stage.get("duration_seconds", 0) for stage in results["stages"].values()
        )

        results["summary"] = {
            "successful_stages": successful_stages,
            "total_stages": total_stages,
            "success_rate": successful_stages / total_stages if total_stages > 0 else 0,
            "total_duration_seconds": total_duration,
            "pipeline_status": "success" if successful_stages == total_stages else "partial",
            "major_failures": [
                name for name, stage in results["stages"].items() if stage.get("status") == "failed"
            ],
        }

        print(f"\n📈 Module Summary:")
        print(
            f"   Success Rate: {results['summary']['success_rate']:.1%} ({successful_stages}/{total_stages})"
        )
        print(f"   Total Duration: {total_duration:.2f}s")
        print(f"   Status: {results['summary']['pipeline_status']}")
        if results["summary"]["major_failures"]:
            print(f"   Failed stages: {', '.join(results['summary']['major_failures'])}")

        return results

    def run_all_tests(self):
        """Run comprehensive tests on all mathlib4 modules."""
        modules = [
            ("List_Basic.lean", "Mathlib/Data/List/Basic"),
            ("Group_Basic.lean", "Mathlib/Algebra/Group/Basic"),
            ("Logic_Basic.lean", "Mathlib/Logic/Basic"),
            ("Nat_Basic.lean", "Mathlib/Data/Nat/Basic"),
            ("Order_Basic.lean", "Mathlib/Order/Basic"),
        ]

        print("🚀 COMPREHENSIVE SIMPULSE PIPELINE TEST")
        print("=" * 80)
        print(f"📅 Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Testing {len(modules)} real mathlib4 modules")
        print(f"📋 Pipeline stages: Rule Extraction → Analysis → Pattern Detection → Optimization")

        all_results = []

        for filename, module_name in modules:
            module_path = Path(__file__).parent / filename

            if module_path.exists():
                result = self.test_single_module(module_path, module_name)
                all_results.append(result)
            else:
                print(f"❌ Module not found: {filename}")

        # Generate comprehensive report
        self.generate_final_report(all_results)
        return all_results

    def generate_final_report(self, all_results):
        """Generate comprehensive final report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(__file__).parent / f"mathlib4_comprehensive_report_{timestamp}.json"

        # Calculate aggregate statistics
        aggregate_stats = {
            "test_overview": {
                "total_modules": len(all_results),
                "test_timestamp": datetime.now().isoformat(),
                "simpulse_version": "Phase 3 Milestone 3.2",
                "test_type": "Real Mathlib4 Production Code Analysis",
            }
        }

        if all_results:
            # Aggregate basic statistics
            total_lines = sum(
                r["basic_stats"]["total_lines"] for r in all_results if "basic_stats" in r
            )
            total_theorems = sum(
                r["basic_stats"]["theorem_count"] for r in all_results if "basic_stats" in r
            )
            total_simp_annotations = sum(
                r["basic_stats"]["simp_annotations"] for r in all_results if "basic_stats" in r
            )

            # Calculate success rates by stage
            stage_success_rates = {}
            stage_names = [
                "rule_extraction",
                "traditional_analyzer",
                "pattern_analysis",
                "smart_optimization",
            ]

            for stage_name in stage_names:
                successful = sum(
                    1
                    for r in all_results
                    if r.get("stages", {}).get(stage_name, {}).get("status") == "success"
                )
                stage_success_rates[stage_name] = {
                    "successful": successful,
                    "total": len(all_results),
                    "success_rate": successful / len(all_results) if all_results else 0,
                }

            # Overall pipeline success
            fully_successful = sum(
                1 for r in all_results if r.get("summary", {}).get("success_rate", 0) == 1.0
            )

            aggregate_stats.update(
                {
                    "content_analysis": {
                        "total_lines_analyzed": total_lines,
                        "total_theorems_found": total_theorems,
                        "total_simp_annotations": total_simp_annotations,
                        "average_file_size": sum(
                            r["basic_stats"]["file_size_bytes"]
                            for r in all_results
                            if "basic_stats" in r
                        )
                        / len(all_results),
                    },
                    "pipeline_performance": {
                        "stage_success_rates": stage_success_rates,
                        "fully_successful_modules": fully_successful,
                        "overall_success_rate": fully_successful / len(all_results),
                        "average_total_duration": sum(
                            r.get("summary", {}).get("total_duration_seconds", 0)
                            for r in all_results
                        )
                        / len(all_results),
                    },
                    "rule_extraction_analysis": {
                        "total_rules_extracted": sum(
                            r.get("stages", {}).get("rule_extraction", {}).get("rules_extracted", 0)
                            for r in all_results
                        ),
                        "extraction_success_rate": stage_success_rates.get(
                            "rule_extraction", {}
                        ).get("success_rate", 0),
                    },
                }
            )

        # Create comprehensive report
        final_report = {
            "aggregate_statistics": aggregate_stats,
            "detailed_results": all_results,
            "conclusions": self.generate_conclusions(all_results),
        }

        # Save report
        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=2)

        # Print executive summary
        self.print_executive_summary(final_report)

        print(f"\n📄 Full report saved: {report_path}")
        return final_report

    def generate_conclusions(self, all_results):
        """Generate honest conclusions about Simpulse performance."""
        if not all_results:
            return {"status": "no_data", "message": "No test results to analyze"}

        # Calculate key metrics
        fully_successful = sum(
            1 for r in all_results if r.get("summary", {}).get("success_rate", 0) == 1.0
        )

        extraction_successful = sum(
            1
            for r in all_results
            if r.get("stages", {}).get("rule_extraction", {}).get("status") == "success"
        )

        total_rules_found = sum(
            r.get("stages", {}).get("rule_extraction", {}).get("rules_extracted", 0)
            for r in all_results
        )

        # Generate honest assessment
        conclusions = {
            "overall_assessment": (
                "production_ready"
                if fully_successful >= len(all_results) * 0.8
                else "needs_improvement"
            ),
            "key_findings": [],
            "strengths": [],
            "limitations": [],
            "recommendations": [],
        }

        # Assess strengths
        if extraction_successful >= len(all_results) * 0.8:
            conclusions["strengths"].append("Rule extraction works reliably on real mathlib4 code")
        if total_rules_found > 0:
            conclusions["strengths"].append(
                f"Successfully extracted {total_rules_found} simp rules from production code"
            )

        # Assess limitations
        failed_modules = [
            r["module_name"]
            for r in all_results
            if r.get("summary", {}).get("success_rate", 0) < 1.0
        ]
        if failed_modules:
            conclusions["limitations"].append(f"Pipeline failed on: {', '.join(failed_modules)}")

        # Generate recommendations
        if fully_successful < len(all_results):
            conclusions["recommendations"].append(
                "Improve error handling for edge cases in mathlib4 code"
            )
        if total_rules_found > 0:
            conclusions["recommendations"].append(
                "Continue development - core functionality proven on real code"
            )

        return conclusions

    def print_executive_summary(self, report):
        """Print executive summary to console."""
        print(f"\n{'='*80}")
        print("📊 EXECUTIVE SUMMARY: SIMPULSE ON REAL MATHLIB4 CODE")
        print(f"{'='*80}")

        stats = report["aggregate_statistics"]

        if "content_analysis" in stats:
            content = stats["content_analysis"]
            performance = stats["pipeline_performance"]

            print(f"\n📋 CONTENT ANALYZED:")
            print(f"   📁 Modules tested: {stats['test_overview']['total_modules']}")
            print(f"   📝 Total lines: {content['total_lines_analyzed']:,}")
            print(f"   🧮 Theorems/lemmas: {content['total_theorems_found']:,}")
            print(f"   ⚡ Simp annotations: {content['total_simp_annotations']:,}")

            print(f"\n🎯 PIPELINE PERFORMANCE:")
            print(
                f"   ✅ Fully successful modules: {performance['fully_successful_modules']}/{stats['test_overview']['total_modules']}"
            )
            print(f"   📈 Overall success rate: {performance['overall_success_rate']:.1%}")
            print(f"   ⏱️ Average duration: {performance['average_total_duration']:.2f}s")

            print(f"\n🔍 STAGE-BY-STAGE RESULTS:")
            stage_names = {
                "rule_extraction": "Rule Extraction",
                "traditional_analyzer": "Traditional Analysis",
                "pattern_analysis": "Pattern Detection",
                "smart_optimization": "Smart Optimization",
            }

            for stage_key, stage_name in stage_names.items():
                stage_data = performance["stage_success_rates"].get(stage_key, {})
                success_rate = stage_data.get("success_rate", 0)
                successful = stage_data.get("successful", 0)
                total = stage_data.get("total", 0)
                print(f"   {stage_name}: {success_rate:.1%} ({successful}/{total})")

        conclusions = report["conclusions"]
        print(f"\n🎯 CONCLUSION: {conclusions['overall_assessment'].upper()}")

        if conclusions["key_findings"]:
            print(f"\n💡 KEY FINDINGS:")
            for finding in conclusions["key_findings"][:3]:
                print(f"   • {finding}")

        if conclusions["strengths"]:
            print(f"\n✅ STRENGTHS:")
            for strength in conclusions["strengths"]:
                print(f"   • {strength}")

        if conclusions["limitations"]:
            print(f"\n⚠️ LIMITATIONS:")
            for limitation in conclusions["limitations"]:
                print(f"   • {limitation}")

        print(
            f"\n🏆 VERDICT: {'Simpulse successfully handles real mathlib4 code!' if conclusions['overall_assessment'] == 'production_ready' else 'Simpulse shows promise but needs refinement for production use.'}"
        )


if __name__ == "__main__":
    test_pipeline = MathlibPipelineTest()
    results = test_pipeline.run_all_tests()
