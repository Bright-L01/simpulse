"""Phase 1 integration example - Claude Code CLI with rule analysis.

This example demonstrates the complete Phase 1 workflow:
1. Extract simp rules from Lean code
2. Profile performance
3. Use Claude Code CLI to suggest optimizations
4. Parse and display suggestions
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simpulse.claude import ClaudeCodeClient, PromptBuilder, ResponseParser
from simpulse.config import Config, load_config
from simpulse.evolution import OptimizationGoal, RuleExtractor
from simpulse.profiling import LeanRunner, TraceParser


async def demonstrate_phase1_workflow():
    """Demonstrate complete Phase 1 workflow."""
    print("Simpulse Phase 1 Integration Demo")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    config.setup_logging()
    
    # Initialize components
    lean_runner = LeanRunner(
        lake_path=config.profiling.lake_path,
        lean_path=config.profiling.lean_path
    )
    
    rule_extractor = RuleExtractor()
    claude_client = ClaudeCodeClient(
        command=config.claude.command_path,
        timeout=config.claude.timeout
    )
    
    trace_parser = TraceParser()
    prompt_builder = PromptBuilder()
    response_parser = ResponseParser()
    
    print("✓ Components initialized")
    
    # Check Claude availability
    if not claude_client._validate_claude_installation():
        print("⚠ Claude Code CLI not available - will simulate responses")
        simulate_claude = True
    else:
        print("✓ Claude Code CLI available")
        simulate_claude = False
    
    # Step 1: Extract rules from a sample file
    print("\n" + "=" * 50)
    print("Step 1: Extract Simp Rules")
    
    # Create a sample Lean file for demonstration
    sample_lean_content = """
-- Sample Lean file with simp rules
namespace Example

@[simp]
theorem list_append_nil (l : List α) : l ++ [] = l := by
  simp [List.append_nil]

@[simp high]
theorem nat_add_zero (n : Nat) : n + 0 = n := by
  simp

@[simp 100]
theorem bool_and_true (b : Bool) : b && true = b := by
  cases b <;> simp

@[simp ↓]
theorem complex_rule (x y : Nat) [DecidableEq α] : 
  (x + y).succ = x.succ + y := by
  simp [Nat.succ_add]

end Example
"""
    
    # Save sample file
    sample_file = Path("sample_rules.lean")
    with open(sample_file, 'w') as f:
        f.write(sample_lean_content)
    
    try:
        # Extract rules
        module_rules = rule_extractor.extract_rules_from_file(sample_file)
        
        print(f"✓ Extracted {len(module_rules.rules)} simp rules")
        
        for rule in module_rules.rules:
            print(f"  - {rule.name} (priority: {rule.priority}, direction: {rule.direction.value})")
        
        # Step 2: Profile performance (simulated since we need a real Lean project)
        print("\n" + "=" * 50)
        print("Step 2: Profile Performance")
        
        # Simulate profiling data
        simulated_profile_data = {
            "total_time_ms": 1250.5,
            "rule_applications": 45,
            "successful_rewrites": 38,
            "failed_rewrites": 7,
            "theorem_usage": {
                "list_append_nil": 15,
                "nat_add_zero": 12,
                "bool_and_true": 8,
                "complex_rule": 3
            },
            "entries": [
                {"name": "Meta.Tactic.simp", "elapsed_ms": 800.2},
                {"name": "type_checking", "elapsed_ms": 250.1},
                {"name": "elaboration", "elapsed_ms": 200.2}
            ]
        }
        
        print("✓ Performance profiling completed (simulated)")
        print(f"  Total time: {simulated_profile_data['total_time_ms']:.1f} ms")
        print(f"  Rule applications: {simulated_profile_data['rule_applications']}")
        print(f"  Success rate: {simulated_profile_data['successful_rewrites']/(simulated_profile_data['successful_rewrites']+simulated_profile_data['failed_rewrites'])*100:.1f}%")
        
        # Step 3: Generate Claude prompt
        print("\n" + "=" * 50)
        print("Step 3: Generate Optimization Suggestions")
        
        prompt = prompt_builder.build_mutation_prompt(
            profile_data=simulated_profile_data,
            rules=module_rules.rules,
            top_k=3,
            context="This is a demonstration of Simpulse Phase 1 integration",
            goal=OptimizationGoal.MINIMIZE_TIME
        )
        
        print("✓ Generated Claude prompt")
        print(f"  Prompt length: {len(prompt)} characters")
        
        # Step 4: Query Claude (or simulate)
        if not simulate_claude:
            print("\n🤖 Querying Claude Code CLI...")
            response = await claude_client.query_claude(prompt)
            
            if response.success:
                print(f"✓ Claude response received ({response.execution_time:.1f}s)")
                claude_output = response.content
            else:
                print(f"✗ Claude query failed: {response.error}")
                simulate_claude = True
        
        if simulate_claude:
            print("\n🤖 Simulating Claude response...")
            claude_output = generate_simulated_claude_response()
            print("✓ Simulated response generated")
        
        # Step 5: Parse suggestions
        print("\n" + "=" * 50)
        print("Step 5: Parse Optimization Suggestions")
        
        suggestions = response_parser.parse_mutations(claude_output)
        
        print(f"✓ Parsed {len(suggestions)} mutation suggestions")
        
        # Display suggestions
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n--- Suggestion {i} ---")
            print(f"Rule: {suggestion.rule_name}")
            print(f"Type: {suggestion.mutation_type.value}")
            print(f"Description: {suggestion.description}")
            print(f"Confidence: {suggestion.confidence:.1%}")
            print(f"Reasoning: {suggestion.reasoning[:100]}...")
            
            if suggestion.estimated_impact:
                print("Estimated Impact:")
                for metric, value in suggestion.estimated_impact.items():
                    print(f"  {metric}: {value}%")
                    
            if suggestion.risks:
                print(f"Risks: {', '.join(suggestion.risks[:2])}...")
        
        # Step 6: Configuration demo
        print("\n" + "=" * 50)
        print("Step 6: Configuration Management")
        
        print(f"Claude backend: {config.claude.backend.value}")
        print(f"Cache enabled: {config.claude.cache_responses}")
        print(f"Optimization goal: {config.optimization.goal.value}")
        print(f"Confidence threshold: {config.optimization.confidence_threshold}")
        
        # Save configuration example
        config_path = Path("example_config.json")
        config.save_to_file(config_path)
        print(f"✓ Sample configuration saved to {config_path}")
        
        print("\n" + "=" * 50)
        print("✅ Phase 1 integration demo completed successfully!")
        print("\nNext steps:")
        print("1. Test with real Lean project")
        print("2. Implement mutation application (Phase 2)")
        print("3. Add evolutionary optimization (Phase 3)")
        
    finally:
        # Cleanup
        if sample_file.exists():
            sample_file.unlink()
        
        config_path = Path("example_config.json")
        if config_path.exists():
            config_path.unlink()


def generate_simulated_claude_response() -> str:
    """Generate a simulated Claude response for testing."""
    return '''Based on the performance profile and simp rules analysis, here are my top optimization suggestions:

```json
{
  "suggestions": [
    {
      "rule_name": "complex_rule",
      "mutation_type": "priority_change",
      "description": "Lower priority of rarely used complex rule",
      "original_declaration": "@[simp ↓] theorem complex_rule (x y : Nat) [DecidableEq α] : (x + y).succ = x.succ + y",
      "mutated_declaration": "@[simp 50] theorem complex_rule (x y : Nat) [DecidableEq α] : (x + y).succ = x.succ + y",
      "reasoning": "This rule is only used 3 times but has high cost due to backward direction. Lowering priority and removing backward direction should improve performance.",
      "confidence": 85,
      "estimated_impact": {
        "time_improvement_percent": 12,
        "memory_impact_percent": 5,
        "success_rate_change": 0
      },
      "risks": ["May affect proof automation in edge cases"],
      "prerequisites": ["Verify no dependent proofs break"]
    },
    {
      "rule_name": "list_append_nil",
      "mutation_type": "priority_change", 
      "description": "Increase priority of frequently used rule",
      "original_declaration": "@[simp] theorem list_append_nil (l : List α) : l ++ [] = l",
      "mutated_declaration": "@[simp high] theorem list_append_nil (l : List α) : l ++ [] = l",
      "reasoning": "This rule is used 15 times (most frequent). Higher priority will make simp try it earlier, potentially reducing search time.",
      "confidence": 78,
      "estimated_impact": {
        "time_improvement_percent": 8,
        "memory_impact_percent": 0,
        "success_rate_change": 2
      },
      "risks": ["May interfere with other list simplification rules"],
      "prerequisites": ["Test with comprehensive list operation suite"]
    },
    {
      "rule_name": "bool_and_true",
      "mutation_type": "condition_add",
      "description": "Add decidability condition to improve matching",
      "original_declaration": "@[simp 100] theorem bool_and_true (b : Bool) : b && true = b",
      "mutated_declaration": "@[simp 100] theorem bool_and_true (b : Bool) [Decidable (b = true)] : b && true = b", 
      "reasoning": "Adding decidability condition can help simp make better decisions about when to apply this rule, reducing failed attempts.",
      "confidence": 65,
      "estimated_impact": {
        "time_improvement_percent": 5,
        "memory_impact_percent": 2,
        "success_rate_change": 3
      },
      "risks": ["May make rule less generally applicable", "Could break existing proofs"],
      "prerequisites": ["Verify decidability instances are available"]
    }
  ],
  "summary": {
    "total_suggestions": 3,
    "expected_total_improvement": "15-25%",
    "confidence_average": 76,
    "recommendations": [
      "Focus on optimizing frequently used rules first",
      "Consider rule interaction patterns when changing priorities",
      "Always validate changes with comprehensive test suite"
    ]
  }
}
```

The analysis shows that rule usage frequency and complexity are the main factors affecting performance. The suggested mutations target these areas while maintaining correctness.'''


async def main():
    """Main function."""
    await demonstrate_phase1_workflow()


if __name__ == "__main__":
    asyncio.run(main())