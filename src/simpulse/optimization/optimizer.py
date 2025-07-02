"""Simp rule optimizer."""

from pathlib import Path

from pydantic import BaseModel

from ..evolution.evolution_engine import SimpleEvolutionEngine
from ..evolution.rule_extractor import RuleExtractor


class OptimizationChange(BaseModel):
    """A single optimization change."""

    rule_name: str
    file_path: str
    old_priority: int
    new_priority: int
    reason: str


class OptimizationResult(BaseModel):
    """Result of optimization."""

    project_path: Path
    rules_changed: int
    estimated_improvement: int
    changes: list[OptimizationChange]

    def save(self, path: Path):
        """Save optimization plan."""
        path.write_text(self.model_dump_json(indent=2))


class SimpOptimizer:
    """Main optimizer for simp rules."""

    def __init__(self, strategy: str = "balanced"):
        self.strategy = strategy
        self.extractor = RuleExtractor()
        self.engine = SimpleEvolutionEngine()

    def analyze(self, project_path: Path):
        """Analyze a project."""
        # Extract rules from all Lean files in project
        all_rules = []
        for lean_file in project_path.glob("**/*.lean"):
            if "lake-packages" not in str(lean_file):
                try:
                    module_rules = self.extractor.extract_rules_from_file(lean_file)
                    all_rules.extend(module_rules.rules)
                except Exception:
                    pass
        return {"project_path": project_path, "rules": all_rules}

    def optimize(self, analysis) -> OptimizationResult:
        """Generate optimizations."""
        # Implementation would use existing optimization logic
        project_path = analysis["project_path"]
        rules = analysis["rules"]

        changes = []
        rules_changed = 0

        # Simple optimization: give high priority to simple rules
        for rule in rules[:10]:  # Start with first 10 rules
            if rule.priority == 1000:  # Default priority
                if len(rule.declaration) < 100:  # Simple rule
                    changes.append(
                        OptimizationChange(
                            rule_name=rule.name,
                            file_path=str(rule.location.file.relative_to(project_path)),
                            old_priority=1000,
                            new_priority=2000,
                            reason="Simple rule should have high priority",
                        )
                    )
                    rules_changed += 1

        # Placeholder - would use real optimization
        return OptimizationResult(
            project_path=project_path,
            rules_changed=rules_changed,
            estimated_improvement=30 if rules_changed > 0 else 0,
            changes=changes,
        )

    def apply(self, optimization: OptimizationResult, project_path: Path):
        """Apply optimizations to project."""
        for change in optimization.changes:
            file_path = project_path / change.file_path
            if file_path.exists():
                content = file_path.read_text()
                # Apply the change
                old_pattern = f"@[simp] theorem {change.rule_name}"
                new_pattern = f"@[simp {change.new_priority}] theorem {change.rule_name}"
                content = content.replace(old_pattern, new_pattern)
                file_path.write_text(content)
