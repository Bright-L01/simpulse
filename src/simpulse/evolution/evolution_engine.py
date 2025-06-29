"""Minimal evolution engine for simp optimization."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..profiling.lean_runner import LeanRunner
from .mutation_applicator import MutationApplicator
from .rule_extractor import RuleExtractor


@dataclass
class OptimizationResult:
    """Result of optimization attempt."""
    improved: bool
    improvement_percent: float = 0.0
    best_mutation: Optional[str] = None
    baseline_time: float = 0.0
    optimized_time: float = 0.0


class SimpleEvolutionEngine:
    """Minimal engine that just tries priority swaps."""
    
    def __init__(self):
        self.extractor = RuleExtractor()
        self.applicator = MutationApplicator()
        self.runner = LeanRunner()
        
    async def optimize_file(self, lean_file: Path) -> OptimizationResult:
        """Try simple priority optimizations."""
        
        # Extract rules
        module_rules = self.extractor.extract_rules_from_file(lean_file)
        rules = module_rules.rules
        
        if len(rules) < 2:
            return OptimizationResult(improved=False)
            
        # Profile baseline
        baseline = await self.runner.profile_file(lean_file)
        baseline_time = baseline.get("total_time", 0)
        
        # Try swapping priorities of rule pairs
        best_time = baseline_time
        best_mutation = None
        
        for i in range(min(len(rules), 5)):
            for j in range(i + 1, min(len(rules), 5)):
                # Create swap mutation
                mutation = f"Swap priorities: {rules[i].name} <-> {rules[j].name}"
                
                # Apply and test
                try:
                    mutated_file = self.applicator.apply_simple_swap(
                        lean_file, rules[i], rules[j]
                    )
                    
                    result = await self.runner.profile_file(mutated_file)
                    time = result.get("total_time", float('inf'))
                    
                    if time < best_time:
                        best_time = time
                        best_mutation = mutation
                        
                except Exception:
                    continue
                    
        # Calculate improvement
        if best_mutation and best_time < baseline_time:
            improvement = (baseline_time - best_time) / baseline_time * 100
            return OptimizationResult(
                improved=True,
                improvement_percent=improvement,
                best_mutation=best_mutation,
                baseline_time=baseline_time,
                optimized_time=best_time
            )
        else:
            return OptimizationResult(
                improved=False,
                baseline_time=baseline_time,
                optimized_time=baseline_time
            )
