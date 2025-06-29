"""Data models for Simpulse evolution and optimization.

This module defines core data structures for representing simp rules,
mutations, performance metrics, and optimization results.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class SimpPriority(Enum):
    """Simp rule priority levels."""
    HIGH = "high"
    DEFAULT = "default"
    LOW = "low"
    NUMERIC = "numeric"  # Custom numeric priority


class SimpDirection(Enum):
    """Simp rule direction."""
    FORWARD = "forward"    # Normal direction
    BACKWARD = "backward"  # ↓ attribute


class MutationType(Enum):
    """Types of mutations that can be applied to simp rules."""
    PRIORITY_CHANGE = "priority_change"
    CONDITION_ADD = "condition_add"
    CONDITION_REMOVE = "condition_remove"
    CONDITION_MODIFY = "condition_modify"
    PATTERN_SIMPLIFY = "pattern_simplify"
    PATTERN_GENERALIZE = "pattern_generalize"
    DIRECTION_CHANGE = "direction_change"
    RULE_SPLIT = "rule_split"
    RULE_COMBINE = "rule_combine"
    RULE_DISABLE = "rule_disable"


class OptimizationGoal(Enum):
    """Optimization objectives."""
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_MEMORY = "minimize_memory"
    MINIMIZE_STEPS = "minimize_steps"
    MAXIMIZE_SUCCESS_RATE = "maximize_success_rate"
    BALANCE_ALL = "balance_all"


@dataclass
class SourceLocation:
    """Location information for a simp rule in source code."""
    file: Path
    line: int
    column: int = 0
    module: Optional[str] = None
    
    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"


@dataclass
class SimpRule:
    """Represents a single simp rule with metadata."""
    name: str
    declaration: str  # Full Lean declaration
    priority: Union[int, SimpPriority] = SimpPriority.DEFAULT
    direction: SimpDirection = SimpDirection.FORWARD
    location: Optional[SourceLocation] = None
    conditions: List[str] = field(default_factory=list)
    pattern: Optional[str] = None
    rhs: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def priority_numeric(self) -> int:
        """Get numeric priority value."""
        if isinstance(self.priority, int):
            return self.priority
        elif self.priority == SimpPriority.HIGH:
            return 1000
        elif self.priority == SimpPriority.LOW:
            return 1
        else:  # DEFAULT
            return 500
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "declaration": self.declaration,
            "priority": self.priority.value if isinstance(self.priority, SimpPriority) else self.priority,
            "direction": self.direction.value,
            "location": str(self.location) if self.location else None,
            "conditions": self.conditions,
            "pattern": self.pattern,
            "rhs": self.rhs,
            "metadata": self.metadata
        }


@dataclass
class ModuleRules:
    """Collection of simp rules from a module."""
    module_name: str
    file_path: Path
    rules: List[SimpRule]
    imports: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_rules_by_priority(self) -> Dict[str, List[SimpRule]]:
        """Group rules by priority level."""
        groups = {"high": [], "default": [], "low": [], "numeric": []}
        
        for rule in self.rules:
            if isinstance(rule.priority, int):
                groups["numeric"].append(rule)
            else:
                groups[rule.priority.value].append(rule)
                
        return groups
        
    def get_total_rules(self) -> int:
        """Get total number of rules."""
        return len(self.rules)


@dataclass
class MutationSuggestion:
    """Represents a suggested mutation to a simp rule."""
    rule_name: str
    mutation_type: MutationType
    description: str
    original_declaration: str
    mutated_declaration: str
    reasoning: str
    confidence: float  # 0.0 to 1.0
    estimated_impact: Dict[str, float] = field(default_factory=dict)  # time, memory, etc.
    prerequisites: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule_name": self.rule_name,
            "mutation_type": self.mutation_type.value,
            "description": self.description,
            "original_declaration": self.original_declaration,
            "mutated_declaration": self.mutated_declaration,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "estimated_impact": self.estimated_impact,
            "prerequisites": self.prerequisites,
            "risks": self.risks
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for simp rule evaluation."""
    total_time_ms: float
    rule_applications: int
    successful_rewrites: int
    failed_rewrites: int
    memory_usage_mb: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    steps_count: Optional[int] = None
    theorem_usage: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate rewrite success rate."""
        total = self.successful_rewrites + self.failed_rewrites
        return self.successful_rewrites / total if total > 0 else 0.0
        
    @property
    def avg_time_per_application(self) -> float:
        """Average time per rule application."""
        return self.total_time_ms / self.rule_applications if self.rule_applications > 0 else 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_time_ms": self.total_time_ms,
            "rule_applications": self.rule_applications,
            "successful_rewrites": self.successful_rewrites,
            "failed_rewrites": self.failed_rewrites,
            "memory_usage_mb": self.memory_usage_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "steps_count": self.steps_count,
            "success_rate": self.success_rate,
            "avg_time_per_application": self.avg_time_per_application,
            "theorem_usage": self.theorem_usage
        }


@dataclass
class OptimizationResult:
    """Result of applying a mutation/optimization."""
    original_metrics: PerformanceMetrics
    optimized_metrics: PerformanceMetrics
    mutation: MutationSuggestion
    success: bool
    improvement_ratio: Dict[str, float] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate improvement ratios."""
        if self.success:
            # Calculate time improvement
            if self.original_metrics.total_time_ms > 0:
                time_ratio = (self.original_metrics.total_time_ms - self.optimized_metrics.total_time_ms) / self.original_metrics.total_time_ms
                self.improvement_ratio["time"] = time_ratio
                
            # Calculate memory improvement
            if (self.original_metrics.memory_usage_mb and 
                self.optimized_metrics.memory_usage_mb and
                self.original_metrics.memory_usage_mb > 0):
                memory_ratio = (self.original_metrics.memory_usage_mb - self.optimized_metrics.memory_usage_mb) / self.original_metrics.memory_usage_mb
                self.improvement_ratio["memory"] = memory_ratio
                
            # Calculate success rate improvement
            success_improvement = self.optimized_metrics.success_rate - self.original_metrics.success_rate
            self.improvement_ratio["success_rate"] = success_improvement
            
    @property
    def overall_improvement(self) -> float:
        """Calculate overall improvement score."""
        if not self.success or not self.improvement_ratio:
            return 0.0
            
        # Weighted average of improvements
        weights = {"time": 0.5, "memory": 0.2, "success_rate": 0.3}
        total_score = 0.0
        total_weight = 0.0
        
        for metric, improvement in self.improvement_ratio.items():
            if metric in weights:
                total_score += improvement * weights[metric]
                total_weight += weights[metric]
                
        return total_score / total_weight if total_weight > 0 else 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_metrics": self.original_metrics.to_dict(),
            "optimized_metrics": self.optimized_metrics.to_dict(),
            "mutation": self.mutation.to_dict(),
            "success": self.success,
            "improvement_ratio": self.improvement_ratio,
            "overall_improvement": self.overall_improvement,
            "validation_errors": self.validation_errors,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class OptimizationSession:
    """Represents a complete optimization session."""
    session_id: str
    module_name: str
    goal: OptimizationGoal
    original_rules: List[SimpRule]
    results: List[OptimizationResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    config: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: OptimizationResult):
        """Add optimization result to session."""
        self.results.append(result)
        
    def get_best_results(self, n: int = 5) -> List[OptimizationResult]:
        """Get top n results by improvement."""
        successful_results = [r for r in self.results if r.success]
        return sorted(successful_results, 
                     key=lambda r: r.overall_improvement, 
                     reverse=True)[:n]
                     
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the session."""
        successful = [r for r in self.results if r.success]
        
        if not successful:
            return {
                "total_attempts": len(self.results),
                "successful": 0,
                "success_rate": 0.0,
                "best_improvement": 0.0,
                "avg_improvement": 0.0
            }
            
        improvements = [r.overall_improvement for r in successful]
        
        return {
            "total_attempts": len(self.results),
            "successful": len(successful),
            "success_rate": len(successful) / len(self.results),
            "best_improvement": max(improvements),
            "avg_improvement": sum(improvements) / len(improvements),
            "duration_minutes": (self.end_time - self.start_time).total_seconds() / 60 if self.end_time else None
        }
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "module_name": self.module_name,
            "goal": self.goal.value,
            "original_rules": [rule.to_dict() for rule in self.original_rules],
            "results": [result.to_dict() for result in self.results],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "config": self.config,
            "summary": self.get_session_summary()
        }
        
    def save_to_file(self, file_path: Path):
        """Save session to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load_from_file(cls, file_path: Path) -> 'OptimizationSession':
        """Load session from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Reconstruct objects from dictionaries
        # This is a simplified version - full implementation would need
        # proper deserialization of all nested objects
        session = cls(
            session_id=data["session_id"],
            module_name=data["module_name"],
            goal=OptimizationGoal(data["goal"]),
            original_rules=[],  # Would need full deserialization
            start_time=datetime.fromisoformat(data["start_time"]),
            config=data["config"]
        )
        
        if data["end_time"]:
            session.end_time = datetime.fromisoformat(data["end_time"])
            
        return session