#!/usr/bin/env python3
"""Simple test of rule extraction on mathlib4 modules."""

import sys
from pathlib import Path

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simpulse.analyzer import LeanAnalyzer
from simpulse.evolution.rule_extractor import RuleExtractor


def test_rule_extraction():
    """Test rule extraction on List_Basic.lean."""
    extractor = RuleExtractor()
    analyzer = LeanAnalyzer()

    # Test on List_Basic.lean first
    list_file = Path(__file__).parent / "List_Basic.lean"

    print(f"Testing rule extraction on {list_file}")
    print(f"File exists: {list_file.exists()}")
    print(f"File size: {list_file.stat().st_size} bytes")

    try:
        # Extract using rule extractor
        result = extractor.extract_rules_from_file(list_file)
        print(f"✅ Rule extraction successful")
        print(f"📊 Rules found: {len(result.rules)}")
        print(f"📁 Module: {result.module_name}")

        # Show first few rules
        for i, rule in enumerate(result.rules[:5]):
            line_info = f"line {rule.location.line}" if rule.location else "unknown line"
            print(f"   {i+1}. {rule.name} ({line_info})")

        # Check metadata for any issues
        if "extraction_errors" in result.metadata:
            errors = result.metadata["extraction_errors"]
            print(f"⚠️ Errors encountered: {len(errors)}")
            for error in errors[:3]:
                print(f"   - {error}")

    except Exception as e:
        print(f"❌ Rule extraction failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        # Test with analyzer too
        analysis = analyzer.analyze_file(list_file)
        print(f"✅ Analyzer successful: {analysis.syntax_valid}")
        print(f"📊 Analyzer found {len(analysis.simp_rules)} simp rules")

    except Exception as e:
        print(f"❌ Analyzer failed: {e}")


if __name__ == "__main__":
    test_rule_extraction()
