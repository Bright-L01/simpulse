from dataclasses import dataclass
from typing import Any, Dict, List

from simpulse.evolution import OptimizationGoal, SimpRule


@dataclass
class PromptBuilder:
    """Builds prompts for Claude Code CLI."""

    def build_mutation_prompt(
        self,
        profile_data: Dict[str, Any],
        rules: List[SimpRule],
        top_k: int,
        context: str,
        goal: OptimizationGoal,
    ) -> str:
        """Builds a prompt for Claude to suggest simp rule mutations."""
        # This is a placeholder implementation.
        # A real implementation would construct a detailed prompt
        # based on the provided data.
        return (
            f"Analyze the following Lean 4 simp rule performance data and suggest {top_k} "
            f"mutations to optimize for {goal.value}.\n\n"
            f"Performance Data: {profile_data}\n\n"
            f"Simp Rules: {rules}\n\n"
            f"Context: {context}"
        )
