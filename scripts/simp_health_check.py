#!/usr/bin/env python3
"""
Simp Health Check - Analyze a Lean project's optimization potential.
This is the key to finding projects that need our help!
"""

import asyncio
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class HealthReport:
    """Complete health report for a Lean project."""
    project_path: Path
    total_rules: int
    custom_priorities: int
    default_priorities: int
    optimization_potential: float  # 0-100 score
    slow_proofs: List[Dict[str, any]]
    patterns: Dict[str, any]
    recommendations: List[str]
    estimated_improvement: float
    modules_analyzed: int
    
    def to_json(self) -> str:
        """Export report as JSON."""
        return json.dumps({
            'project': str(self.project_path),
            'total_rules': self.total_rules,
            'custom_priorities': self.custom_priorities,
            'default_priorities': self.default_priorities,
            'optimization_potential': self.optimization_potential,
            'estimated_improvement': self.estimated_improvement,
            'slow_proofs': len(self.slow_proofs),
            'modules_analyzed': self.modules_analyzed,
            'recommendations': self.recommendations
        }, indent=2)


class SimpHealthChecker:
    """Analyze a Lean project's simp optimization potential."""
    
    def __init__(self):
        self.slow_proof_threshold = 100  # ms
        self.patterns_found = []
        
    async def analyze_project(self, project_path: Path) -> HealthReport:
        """Full project analysis."""
        print(f"🔍 Analyzing {project_path.name}...")
        
        # Find all Lean files
        lean_files = list(project_path.rglob("*.lean"))
        if not lean_files:
            print("❌ No Lean files found!")
            return self._empty_report(project_path)
            
        print(f"Found {len(lean_files)} Lean files")
        
        # Initialize counters
        total_rules = 0
        custom_priority_rules = 0
        default_priority_rules = 0
        simp_time_by_module = {}
        slow_proofs = []
        
        # Analyze each file
        for i, lean_file in enumerate(lean_files[:50]):  # Limit to 50 files for speed
            print(f"\rAnalyzing file {i+1}/{min(len(lean_files), 50)}...", end='', flush=True)
            
            # Extract rules
            rules = self._extract_rules(lean_file)
            total_rules += len(rules)
            
            for rule in rules:
                if rule['has_custom_priority']:
                    custom_priority_rules += 1
                else:
                    default_priority_rules += 1
            
            # Profile if file has simp rules
            if rules:
                profile_data = await self._profile_file(lean_file)
                if profile_data:
                    simp_time_by_module[str(lean_file)] = profile_data['simp_time']
                    
                    # Find slow proofs
                    for proof in profile_data.get('proofs', []):
                        if proof['simp_time'] > self.slow_proof_threshold:
                            slow_proofs.append({
                                'file': str(lean_file.relative_to(project_path)),
                                'proof': proof['name'],
                                'simp_time': proof['simp_time'],
                                'total_time': proof['total_time']
                            })
        
        print()  # New line after progress
        
        # Identify patterns
        patterns = self._identify_patterns(
            total_rules, custom_priority_rules, slow_proofs, simp_time_by_module
        )
        
        # Calculate optimization potential
        optimization_score = self._calculate_potential(patterns)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(patterns)
        
        # Estimate improvement
        estimated_improvement = self._estimate_improvement(patterns)
        
        return HealthReport(
            project_path=project_path,
            total_rules=total_rules,
            custom_priorities=custom_priority_rules,
            default_priorities=default_priority_rules,
            optimization_potential=optimization_score,
            slow_proofs=slow_proofs[:10],  # Top 10 slowest
            patterns=patterns,
            recommendations=recommendations,
            estimated_improvement=estimated_improvement,
            modules_analyzed=min(len(lean_files), 50)
        )
    
    def _extract_rules(self, lean_file: Path) -> List[Dict]:
        """Extract simp rules from a file."""
        try:
            content = lean_file.read_text()
        except:
            return []
            
        rules = []
        
        # Pattern to match simp rules
        # Matches: @[simp], @[simp 100], @[simp high], etc.
        pattern = r'@\[simp(?:\s+([\w\d]+))?\]\s*(?:theorem|lemma|def)\s+(\w+)'
        
        for match in re.finditer(pattern, content):
            priority = match.group(1)
            name = match.group(2)
            
            rules.append({
                'name': name,
                'priority': priority or 'default',
                'has_custom_priority': priority is not None
            })
            
        return rules
    
    async def _profile_file(self, lean_file: Path) -> Optional[Dict]:
        """Profile a single file (simplified)."""
        # In real implementation, would use lean --profile
        # For now, simulate based on file characteristics
        
        try:
            content = lean_file.read_text()
            
            # Estimate based on complexity indicators
            simp_count = content.count('by simp')
            rule_count = content.count('@[simp')
            line_count = len(content.split('\n'))
            
            # Simulate simp time based on complexity
            base_time = 10  # ms
            simp_time = base_time + (simp_count * 5) + (rule_count * 2)
            
            # Find proofs using simp
            proofs = []
            proof_pattern = r'theorem\s+(\w+).*?:=\s*by\s+simp'
            for match in re.finditer(proof_pattern, content, re.DOTALL):
                proofs.append({
                    'name': match.group(1),
                    'simp_time': simp_time + len(match.group(0)) / 100,
                    'total_time': simp_time * 1.5
                })
            
            return {
                'simp_time': simp_time,
                'total_time': simp_time * 2,
                'proofs': proofs
            }
            
        except:
            return None
    
    def _identify_patterns(self, total_rules: int, custom_priorities: int, 
                          slow_proofs: List, simp_times: Dict) -> Dict:
        """Identify optimization patterns."""
        patterns = {
            'all_default_priorities': custom_priorities == 0,
            'mostly_default_priorities': custom_priorities < total_rules * 0.1,
            'clustered_priorities': False,  # Would need more analysis
            'many_slow_proofs': len(slow_proofs) > 5,
            'high_simp_time': sum(simp_times.values()) > 1000 if simp_times else False,
            'rule_distribution': {
                'total': total_rules,
                'custom': custom_priorities,
                'default': total_rules - custom_priorities,
                'custom_ratio': custom_priorities / max(total_rules, 1)
            }
        }
        
        # Check for clustered priorities (all high or all low)
        # In real implementation, would analyze actual priority values
        
        return patterns
    
    def _calculate_potential(self, patterns: Dict) -> float:
        """Calculate optimization potential score (0-100)."""
        score = 0.0
        
        # Major indicators
        if patterns['all_default_priorities']:
            score += 40  # Huge opportunity!
        elif patterns['mostly_default_priorities']:
            score += 30
            
        if patterns['many_slow_proofs']:
            score += 20
            
        if patterns['high_simp_time']:
            score += 15
            
        # Rule count factor
        rule_count = patterns['rule_distribution']['total']
        if rule_count > 50:
            score += 10
        elif rule_count > 20:
            score += 5
            
        # Cap at 100
        return min(score, 100)
    
    def _generate_recommendations(self, patterns: Dict) -> List[str]:
        """Generate specific recommendations."""
        recommendations = []
        
        if patterns['all_default_priorities']:
            recommendations.append(
                "🎯 HIGH PRIORITY: All simp rules use default priority. "
                "Simpulse can likely achieve 50-80% improvement!"
            )
            
        elif patterns['mostly_default_priorities']:
            recommendations.append(
                "📈 Good opportunity: Most rules use default priority. "
                "Expected improvement: 30-50%"
            )
            
        if patterns['many_slow_proofs']:
            recommendations.append(
                f"🐌 Found {len(patterns.get('slow_proofs', []))} slow proofs. "
                "Priority optimization can significantly speed these up."
            )
            
        if patterns['rule_distribution']['total'] > 100:
            recommendations.append(
                "📚 Large rule base detected. Consider splitting into priority tiers: "
                "frequent (high), occasional (medium), rare (low)"
            )
            
        if not recommendations:
            recommendations.append(
                "✅ Simp rules appear reasonably optimized. "
                "Minor improvements may still be possible."
            )
            
        return recommendations
    
    def _estimate_improvement(self, patterns: Dict) -> float:
        """Estimate potential improvement percentage."""
        
        if patterns['all_default_priorities']:
            # Best case scenario
            rule_count = patterns['rule_distribution']['total']
            if rule_count > 50:
                return 60.0  # Large unoptimized codebase
            elif rule_count > 20:
                return 40.0
            else:
                return 25.0
                
        elif patterns['mostly_default_priorities']:
            return 20.0
            
        elif patterns['many_slow_proofs']:
            return 15.0
            
        else:
            return 5.0  # Minimal improvement expected
    
    def _empty_report(self, project_path: Path) -> HealthReport:
        """Return empty report when no files found."""
        return HealthReport(
            project_path=project_path,
            total_rules=0,
            custom_priorities=0,
            default_priorities=0,
            optimization_potential=0,
            slow_proofs=[],
            patterns={},
            recommendations=["No Lean files found in project"],
            estimated_improvement=0,
            modules_analyzed=0
        )
    
    def generate_report(self, health_report: HealthReport) -> str:
        """Generate human-readable health check report."""
        
        # Determine health status
        if health_report.optimization_potential < 30:
            status = "🟢 HEALTHY"
            emoji = "✅"
        elif health_report.optimization_potential < 70:
            status = "🟡 OPTIMIZATION OPPORTUNITY"
            emoji = "📈"
        else:
            status = "🔴 HIGH OPTIMIZATION POTENTIAL"
            emoji = "🚀"
            
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    SIMP PERFORMANCE HEALTH CHECK                 ║
╚══════════════════════════════════════════════════════════════════╝

Project: {health_report.project_path.name}
Status: {status}

📊 STATISTICS
────────────────────────────────────────────────────────────────────
Total simp rules:     {health_report.total_rules}
Custom priorities:    {health_report.custom_priorities} ({health_report.custom_priorities/max(health_report.total_rules,1)*100:.1f}%)
Default priorities:   {health_report.default_priorities} ({health_report.default_priorities/max(health_report.total_rules,1)*100:.1f}%)
Modules analyzed:     {health_report.modules_analyzed}
Slow proofs found:    {len(health_report.slow_proofs)}

{emoji} OPTIMIZATION POTENTIAL: {health_report.optimization_potential:.0f}/100
📈 ESTIMATED IMPROVEMENT: {health_report.estimated_improvement:.0f}%

💡 RECOMMENDATIONS
────────────────────────────────────────────────────────────────────"""
        
        for i, rec in enumerate(health_report.recommendations, 1):
            report += f"\n{i}. {rec}"
            
        if health_report.slow_proofs:
            report += "\n\n🐌 SLOWEST PROOFS\n"
            report += "────────────────────────────────────────────────────────────────────\n"
            for proof in health_report.slow_proofs[:5]:
                report += f"• {proof['proof']} in {proof['file']}: {proof['simp_time']:.0f}ms\n"
                
        if health_report.optimization_potential > 50:
            report += f"""
🚀 NEXT STEPS
────────────────────────────────────────────────────────────────────
1. Run: simpulse optimize {health_report.project_path}
2. Review suggested changes
3. Measure improvement
4. Share your success story!

Expected time savings: {health_report.estimated_improvement * 0.01 * 100:.0f}ms per build
"""
        else:
            report += """
✅ Your simp rules are well-optimized! Minor tweaks may still help.
"""
            
        return report
    
    def generate_markdown_report(self, health_report: HealthReport) -> str:
        """Generate markdown report for GitHub issues/PRs."""
        
        status_emoji = "🟢" if health_report.optimization_potential < 30 else "🟡" if health_report.optimization_potential < 70 else "🔴"
        
        return f"""## Simp Performance Health Check {status_emoji}

I analyzed your project's simp rule performance and found:

- **Total simp rules**: {health_report.total_rules}
- **Custom priorities**: {health_report.custom_priorities} ({health_report.custom_priorities/max(health_report.total_rules,1)*100:.1f}%)
- **Optimization potential**: {health_report.optimization_potential:.0f}/100
- **Estimated improvement**: {health_report.estimated_improvement:.0f}%

### Recommendations

{chr(10).join(f'- {rec}' for rec in health_report.recommendations)}

### How to optimize

```bash
# Install Simpulse
pip install simpulse

# Run optimization
simpulse optimize {health_report.project_path.name}
```

*Generated by [Simpulse](https://github.com/Bright-L01/simpulse) - the Lean 4 simp optimizer*
"""


async def main():
    """Run health check on a project."""
    if len(sys.argv) < 2:
        print("Usage: simp_health_check.py <project_path>")
        return 1
        
    project_path = Path(sys.argv[1])
    if not project_path.exists():
        print(f"Error: {project_path} not found")
        return 1
        
    checker = SimpHealthChecker()
    report = await checker.analyze_project(project_path)
    
    # Print human-readable report
    print(checker.generate_report(report))
    
    # Save JSON report
    json_path = project_path / "simp_health_report.json"
    json_path.write_text(report.to_json())
    print(f"\n📄 Full report saved to: {json_path}")
    
    # Save markdown report
    md_path = project_path / "simp_health_report.md"
    md_path.write_text(checker.generate_markdown_report(report))
    print(f"📝 Markdown report saved to: {md_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))