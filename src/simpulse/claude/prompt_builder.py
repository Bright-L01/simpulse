"""Prompt builder for Claude Code CLI interactions.

This module constructs structured prompts for getting intelligent
simp rule optimization suggestions from Claude.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from ..evolution.models import SimpRule, PerformanceMetrics, OptimizationGoal


class PromptBuilder:
    """Builds structured prompts for Claude Code interactions."""
    
    def __init__(self):
        """Initialize prompt builder."""
        self.context_template = self._load_context_template()
        
    def build_mutation_prompt(self,
                             profile_data: Dict[str, Any],
                             rules: List[SimpRule],
                             top_k: int = 5,
                             context: Optional[str] = None,
                             goal: OptimizationGoal = OptimizationGoal.MINIMIZE_TIME) -> str:
        """Build complete mutation analysis prompt.
        
        Args:
            profile_data: Performance profiling data
            rules: List of simp rules to analyze
            top_k: Number of top suggestions to request
            context: Additional context about optimization goals
            goal: Primary optimization objective
            
        Returns:
            Complete prompt string
        """
        sections = []
        
        # Add header and context
        sections.append(self._build_header(goal, context))
        
        # Add performance analysis
        sections.append(self._build_performance_section(profile_data))
        
        # Add rules context
        sections.append(self._build_rules_section(rules))
        
        # Add analysis request
        sections.append(self._build_analysis_request(top_k, goal))
        
        # Add output format specification
        sections.append(self._build_output_format())
        
        return "\n\n".join(sections)
        
    def build_analysis_prompt(self, profile_data: Dict[str, Any]) -> str:
        """Build prompt for performance bottleneck analysis.
        
        Args:
            profile_data: Performance profiling data
            
        Returns:
            Analysis prompt string
        """
        sections = []
        
        sections.append("""# Lean 4 Simp Performance Analysis

I need you to analyze the performance characteristics of simp tactic usage in a Lean 4 module and identify potential bottlenecks and optimization opportunities.""")
        
        sections.append(self._build_performance_section(profile_data))
        
        sections.append("""## Analysis Request

Please provide:

1. **Performance Bottlenecks**: Identify the most time-consuming operations
2. **Optimization Opportunities**: Specific areas where performance could be improved
3. **Rule Usage Patterns**: Analysis of which simp rules are used most frequently
4. **Recommendations**: Concrete suggestions for improvement

Focus on actionable insights that could lead to measurable performance improvements.""")
        
        return "\n\n".join(sections)
        
    def _build_header(self, goal: OptimizationGoal, context: Optional[str]) -> str:
        """Build prompt header with context."""
        header = """# Lean 4 Simp Rule Optimization

I'm working on optimizing simp rules in a Lean 4 codebase using performance profiling data. I need intelligent suggestions for mutations that could improve performance while maintaining correctness."""
        
        goal_descriptions = {
            OptimizationGoal.MINIMIZE_TIME: "minimize total execution time",
            OptimizationGoal.MINIMIZE_MEMORY: "minimize memory usage",
            OptimizationGoal.MINIMIZE_STEPS: "minimize the number of rewrite steps",
            OptimizationGoal.MAXIMIZE_SUCCESS_RATE: "maximize the success rate of simp applications",
            OptimizationGoal.BALANCE_ALL: "balance time, memory, and success rate"
        }
        
        header += f"\n\n**Primary Goal**: {goal_descriptions.get(goal, 'optimize performance')}"
        
        if context:
            header += f"\n\n**Additional Context**: {context}"
            
        return header
        
    def _build_performance_section(self, profile_data: Dict[str, Any]) -> str:
        """Build performance data section."""
        section = "## Performance Profile Data\n"
        
        # Format overall metrics
        if "total_time_ms" in profile_data:
            section += f"- **Total Execution Time**: {profile_data['total_time_ms']:.2f} ms\n"
            
        if "rule_applications" in profile_data:
            section += f"- **Rule Applications**: {profile_data['rule_applications']}\n"
            
        if "successful_rewrites" in profile_data:
            success_rate = 0
            if "failed_rewrites" in profile_data:
                total = profile_data["successful_rewrites"] + profile_data["failed_rewrites"]
                success_rate = profile_data["successful_rewrites"] / total * 100 if total > 0 else 0
            section += f"- **Successful Rewrites**: {profile_data['successful_rewrites']} ({success_rate:.1f}% success rate)\n"
            
        if "failed_rewrites" in profile_data:
            section += f"- **Failed Rewrites**: {profile_data['failed_rewrites']}\n"
            
        # Add theorem usage statistics
        if "theorem_usage" in profile_data and profile_data["theorem_usage"]:
            section += "\n### Most Used Theorems\n"
            theorem_usage = profile_data["theorem_usage"]
            # Sort by usage count
            sorted_theorems = sorted(theorem_usage.items(), key=lambda x: x[1], reverse=True)
            
            for theorem, count in sorted_theorems[:10]:  # Top 10
                section += f"- `{theorem}`: {count} applications\n"
                
        # Add timing breakdown if available
        if "entries" in profile_data and profile_data["entries"]:
            section += "\n### Timing Breakdown\n"
            for entry in profile_data["entries"][:5]:  # Top 5 time consumers
                if isinstance(entry, dict):
                    name = entry.get("name", "unknown")
                    time_ms = entry.get("elapsed_ms", 0)
                    section += f"- `{name}`: {time_ms:.2f} ms\n"
                    
        return section
        
    def _build_rules_section(self, rules: List[SimpRule]) -> str:
        """Build rules context section."""
        section = "## Current Simp Rules\n"
        
        if not rules:
            section += "No simp rules provided for analysis.\n"
            return section
            
        section += f"Analyzing {len(rules)} simp rules:\n\n"
        
        # Group rules by priority
        priority_groups = {"high": [], "default": [], "low": [], "numeric": []}
        
        for rule in rules:
            if isinstance(rule.priority, int):
                priority_groups["numeric"].append(rule)
            else:
                priority_groups[rule.priority.value].append(rule)
                
        # Format each group
        for priority, group_rules in priority_groups.items():
            if not group_rules:
                continue
                
            section += f"### {priority.title()} Priority Rules ({len(group_rules)})\n\n"
            
            for rule in group_rules[:10]:  # Limit to avoid overwhelming
                section += f"**{rule.name}**\n"
                section += f"```lean\n{rule.declaration}\n```\n"
                
                if rule.conditions:
                    section += f"- Conditions: {', '.join(rule.conditions)}\n"
                    
                if rule.location:
                    section += f"- Location: {rule.location}\n"
                    
                section += "\n"
                
        return section
        
    def _build_analysis_request(self, top_k: int, goal: OptimizationGoal) -> str:
        """Build analysis request section."""
        section = f"## Analysis Request\n\nBased on the performance data and simp rules above, please suggest the top {top_k} mutations that could improve performance.\n\n"
        
        section += """For each suggestion, provide:

1. **Rule Name**: Which rule to modify
2. **Mutation Type**: What kind of change (priority adjustment, condition modification, etc.)
3. **Specific Change**: Exact modification to make
4. **Reasoning**: Why this change should improve performance
5. **Estimated Impact**: Predicted performance improvement
6. **Risks**: Potential negative effects or things to watch for
7. **Confidence**: Your confidence level (0-100%) in the suggestion

Focus on mutations that are:
- **Safe**: Unlikely to break correctness
- **Impactful**: Should provide measurable performance improvements
- **Practical**: Can be implemented without major refactoring

Consider these mutation strategies:
- Adjusting rule priorities based on usage patterns
- Adding conditions to make rules more specific
- Simplifying overly complex rules
- Combining frequently used rule patterns
- Disabling rarely used but expensive rules"""

        return section
        
    def _build_output_format(self) -> str:
        """Build output format specification."""
        return """## Output Format

Please structure your response as follows:

```json
{
  "suggestions": [
    {
      "rule_name": "rule_name_here",
      "mutation_type": "priority_change|condition_add|pattern_simplify|etc",
      "description": "Brief description of the change",
      "original_declaration": "original rule text",
      "mutated_declaration": "modified rule text", 
      "reasoning": "Detailed explanation of why this helps",
      "confidence": 85,
      "estimated_impact": {
        "time_improvement_percent": 15,
        "memory_impact_percent": 0,
        "success_rate_change": 0
      },
      "risks": ["potential issue 1", "potential issue 2"],
      "prerequisites": ["requirement 1", "requirement 2"]
    }
  ],
  "summary": {
    "total_suggestions": 5,
    "expected_total_improvement": "10-25%",
    "confidence_average": 78,
    "recommendations": ["general recommendation 1", "general recommendation 2"]
  }
}
```

Ensure the JSON is valid and complete."""
        
    def format_profile_summary(self, profile_data: Dict[str, Any]) -> str:
        """Format profiling data for Claude context.
        
        Args:
            profile_data: Performance profiling data
            
        Returns:
            Formatted summary string
        """
        lines = []
        
        if "total_time_ms" in profile_data:
            lines.append(f"Total Time: {profile_data['total_time_ms']:.2f} ms")
            
        if "rule_applications" in profile_data:
            lines.append(f"Rule Applications: {profile_data['rule_applications']}")
            
        if "successful_rewrites" in profile_data and "failed_rewrites" in profile_data:
            total = profile_data["successful_rewrites"] + profile_data["failed_rewrites"]
            success_rate = profile_data["successful_rewrites"] / total * 100 if total > 0 else 0
            lines.append(f"Success Rate: {success_rate:.1f}%")
            
        return " | ".join(lines)
        
    def format_rules_context(self, rules: List[SimpRule]) -> str:
        """Format simp rules with context.
        
        Args:
            rules: List of simp rules
            
        Returns:
            Formatted rules string
        """
        if not rules:
            return "No rules available"
            
        lines = [f"Total Rules: {len(rules)}"]
        
        # Count by priority
        priority_counts = {}
        for rule in rules:
            if isinstance(rule.priority, int):
                key = f"numeric({rule.priority})"
            else:
                key = rule.priority.value
                
            priority_counts[key] = priority_counts.get(key, 0) + 1
            
        priority_strs = [f"{p}: {c}" for p, c in priority_counts.items()]
        lines.append(f"By Priority: {', '.join(priority_strs)}")
        
        return " | ".join(lines)
        
    def _load_context_template(self) -> str:
        """Load context template for prompts."""
        return """You are an expert in Lean 4 theorem proving and performance optimization. 
You understand simp tactics, rewrite rules, and performance characteristics of automated reasoning in Lean.

Your task is to analyze performance data and suggest intelligent mutations to simp rules that will improve performance while maintaining correctness."""