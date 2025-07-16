"""
Diagnostic Parser for Lean 4.8.0+ Diagnostics Output

Parses real simp performance data from `set_option diagnostics true` output
to provide evidence-based optimization recommendations.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SimpTheoremUsage:
    """Represents usage statistics for a single simp theorem."""
    name: str
    used_count: int = 0
    tried_count: int = 0
    succeeded_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate (used/tried)."""
        if self.tried_count == 0:
            return 0.0
        return self.used_count / self.tried_count
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (higher is better)."""
        if self.tried_count == 0:
            return 0.0
        # Reward high usage and high success rate
        return self.used_count * self.success_rate


@dataclass
class DiagnosticAnalysis:
    """Complete analysis of Lean diagnostic output."""
    simp_theorems: Dict[str, SimpTheoremUsage] = field(default_factory=dict)
    kernel_unfoldings: Dict[str, int] = field(default_factory=dict)
    reduction_unfoldings: Dict[str, int] = field(default_factory=dict)
    total_simp_attempts: int = 0
    looping_theorems: List[str] = field(default_factory=list)
    inefficient_theorems: List[str] = field(default_factory=list)
    
    def get_most_used_theorems(self, limit: int = 10) -> List[SimpTheoremUsage]:
        """Get the most frequently used simp theorems."""
        return sorted(
            self.simp_theorems.values(),
            key=lambda x: x.used_count,
            reverse=True
        )[:limit]
    
    def get_least_efficient_theorems(self, limit: int = 10) -> List[SimpTheoremUsage]:
        """Get theorems with worst success rates (high tried, low used)."""
        inefficient = [
            theorem for theorem in self.simp_theorems.values()
            if theorem.tried_count > 10 and theorem.success_rate < 0.1
        ]
        return sorted(inefficient, key=lambda x: x.tried_count, reverse=True)[:limit]
    
    def detect_looping_patterns(self) -> List[str]:
        """Detect potential looping simp lemmas based on usage patterns."""
        looping = []
        for theorem in self.simp_theorems.values():
            # Heuristic: very high usage count might indicate loops
            if theorem.used_count > 100:
                looping.append(theorem.name)
        return looping


class DiagnosticParser:
    """Parser for Lean 4.8.0+ diagnostic output."""
    
    def __init__(self):
        # Pattern for simp theorem usage: "[simp] used theorems (max: 250, num: 2):"
        self.simp_used_header = re.compile(
            r"\[simp\] used theorems \(max: (\d+), num: (\d+)\):"
        )
        
        # Pattern for simp theorem tries: "[simp] tried theorems (max: 251, num: 2):"
        self.simp_tried_header = re.compile(
            r"\[simp\] tried theorems \(max: (\d+), num: (\d+)\):"
        )
        
        # Pattern for individual theorem usage: "  theorem_name ↦ 250"
        self.theorem_usage = re.compile(
            r"^\s*([^\s↦]+)\s*↦\s*(\d+)(?:,\s*succeeded:\s*(\d+))?", re.MULTILINE
        )
        
        # Pattern for kernel unfoldings: "[kernel] unfolded declarations (max: 56, num: 8):"
        self.kernel_unfolding_header = re.compile(
            r"\[kernel\] unfolded declarations \(max: (\d+), num: (\d+)\):"
        )
        
        # Pattern for reduction unfoldings: "[reduction] unfolded reducible declarations"
        self.reduction_unfolding_header = re.compile(
            r"\[reduction\] unfolded reducible declarations \(max: (\d+), num: (\d+)\):"
        )
    
    def parse_diagnostic_output(self, output: str) -> DiagnosticAnalysis:
        """Parse complete diagnostic output from Lean compilation."""
        analysis = DiagnosticAnalysis()
        
        # Parse simp theorem usage
        self._parse_simp_usage(output, analysis)
        
        # Parse kernel unfoldings
        self._parse_kernel_unfoldings(output, analysis)
        
        # Parse reduction unfoldings
        self._parse_reduction_unfoldings(output, analysis)
        
        # Analyze patterns
        analysis.looping_theorems = analysis.detect_looping_patterns()
        analysis.inefficient_theorems = [
            t.name for t in analysis.get_least_efficient_theorems()
        ]
        
        return analysis
    
    def _parse_simp_usage(self, output: str, analysis: DiagnosticAnalysis) -> None:
        """Parse simp theorem usage statistics."""
        lines = output.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Look for "used theorems" section
            used_match = self.simp_used_header.search(line)
            if used_match:
                max_used = int(used_match.group(1))
                num_theorems = int(used_match.group(2))
                i += 1
                
                # Parse the theorem usage lines (only until we hit another section or empty line)
                while i < len(lines) and lines[i].strip() and not self._is_new_section(lines[i]):
                    usage_line = lines[i]
                    for match in self.theorem_usage.finditer(usage_line):
                        theorem_name = match.group(1)
                        used_count = int(match.group(2))
                        
                        if theorem_name not in analysis.simp_theorems:
                            analysis.simp_theorems[theorem_name] = SimpTheoremUsage(name=theorem_name)
                        
                        analysis.simp_theorems[theorem_name].used_count = used_count
                    i += 1
                continue
            
            # Look for "tried theorems" section
            tried_match = self.simp_tried_header.search(line)
            if tried_match:
                max_tried = int(tried_match.group(1))
                num_theorems = int(tried_match.group(2))
                i += 1
                
                # Parse the theorem tried lines (only until we hit another section or empty line)
                while i < len(lines) and lines[i].strip() and not self._is_new_section(lines[i]):
                    tried_line = lines[i]
                    for match in self.theorem_usage.finditer(tried_line):
                        theorem_name = match.group(1)
                        tried_count = int(match.group(2))
                        succeeded_count = int(match.group(3)) if match.group(3) else tried_count
                        
                        if theorem_name not in analysis.simp_theorems:
                            analysis.simp_theorems[theorem_name] = SimpTheoremUsage(name=theorem_name)
                        
                        theorem = analysis.simp_theorems[theorem_name]
                        theorem.tried_count = tried_count
                        theorem.succeeded_count = succeeded_count
                    i += 1
                continue
            
            i += 1
    
    def _is_new_section(self, line: str) -> bool:
        """Check if a line starts a new diagnostic section."""
        return (line.startswith('[kernel]') or 
                line.startswith('[reduction]') or 
                line.startswith('[simp]'))
    
    def _parse_kernel_unfoldings(self, output: str, analysis: DiagnosticAnalysis) -> None:
        """Parse kernel unfolding statistics."""
        lines = output.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            kernel_match = self.kernel_unfolding_header.search(line)
            if kernel_match:
                i += 1
                
                # Parse the unfolding lines
                while i < len(lines) and lines[i].strip():
                    unfold_line = lines[i]
                    for match in self.theorem_usage.finditer(unfold_line):
                        decl_name = match.group(1)
                        unfold_count = int(match.group(2))
                        analysis.kernel_unfoldings[decl_name] = unfold_count
                    i += 1
                continue
            
            i += 1
    
    def _parse_reduction_unfoldings(self, output: str, analysis: DiagnosticAnalysis) -> None:
        """Parse reduction unfolding statistics."""
        lines = output.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            reduction_match = self.reduction_unfolding_header.search(line)
            if reduction_match:
                i += 1
                
                # Parse the reduction unfolding lines
                while i < len(lines) and lines[i].strip():
                    unfold_line = lines[i]
                    for match in self.theorem_usage.finditer(unfold_line):
                        decl_name = match.group(1)
                        unfold_count = int(match.group(2))
                        analysis.reduction_unfoldings[decl_name] = unfold_count
                    i += 1
                continue
            
            i += 1


class DiagnosticCollector:
    """Collects diagnostic output from Lean compilation."""
    
    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.parser = DiagnosticParser()
    
    def collect_diagnostics(self, files: Optional[List[Path]] = None) -> DiagnosticAnalysis:
        """Collect diagnostic data by running Lean with diagnostics enabled."""
        import subprocess
        import tempfile
        
        if files is None:
            files = list(self.project_path.glob("**/*.lean"))
        
        if not files:
            logger.warning(f"No Lean files found in {self.project_path}")
            return DiagnosticAnalysis()
        
        # Create a temporary Lean file that imports all target files and enables diagnostics
        temp_content = self._create_diagnostic_test_file(files)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as temp_file:
            temp_file.write(temp_content)
            temp_file_path = Path(temp_file.name)
        
        try:
            # Run Lean with the diagnostic test file (no --check flag in newer versions)
            result = subprocess.run(
                ['lean', str(temp_file_path)],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Parse the diagnostic output
            full_output = result.stdout + result.stderr
            return self.parser.parse_diagnostic_output(full_output)
            
        except subprocess.TimeoutExpired:
            logger.error("Diagnostic collection timed out after 5 minutes")
            return DiagnosticAnalysis()
        except subprocess.CalledProcessError as e:
            logger.error(f"Lean compilation failed: {e}")
            return DiagnosticAnalysis()
        finally:
            # Clean up temporary file
            temp_file_path.unlink(missing_ok=True)
    
    def _create_diagnostic_test_file(self, files: List[Path]) -> str:
        """Create a Lean file that enables diagnostics and tests all target files."""
        content = []
        content.append("-- Diagnostic collection file")
        content.append("set_option diagnostics true")
        content.append("set_option diagnostics.threshold 1")
        content.append("")
        
        # Add imports for all files (convert paths to proper module names)
        for file_path in files:
            try:
                # Convert file path to module name
                relative_path = file_path.relative_to(self.project_path)
                module_parts = []
                
                # Handle nested directory structure
                for part in relative_path.parts[:-1]:  # All parts except filename
                    module_parts.append(part)
                
                # Add filename without extension
                module_parts.append(relative_path.stem)
                
                module_name = '.'.join(module_parts)
                content.append(f"import {module_name}")
            except ValueError:
                # Skip files outside project path
                continue
        
        content.append("")
        content.append("-- Trigger some simp usage to generate diagnostics")
        content.append("example : 1 + 1 = 2 := by simp")
        content.append("example (n : Nat) : n + 0 = n := by simp")
        content.append("example (l : List α) : l ++ [] = l := by simp")
        
        return '\n'.join(content)


def analyze_project(project_path: str) -> DiagnosticAnalysis:
    """Main entry point for diagnostic analysis."""
    collector = DiagnosticCollector(Path(project_path))
    return collector.collect_diagnostics()


# Example usage and testing
if __name__ == "__main__":
    # Test with sample diagnostic output
    sample_output = """
[simp] used theorems (max: 250, num: 2):
  one_plus_eq_plus_one ↦ 250
  plus_one_eq_one_plus ↦ 250
[simp] tried theorems (max: 251, num: 2):
  plus_one_eq_one_plus ↦ 251, succeeded: 250
  one_plus_eq_plus_one ↦ 250, succeeded: 250
[kernel] unfolded declarations (max: 56, num: 3):
  List.rec ↦ 56
  PProd.fst ↦ 48
  List.casesOn ↦ 29
"""
    
    parser = DiagnosticParser()
    analysis = parser.parse_diagnostic_output(sample_output)
    
    print(f"Parsed {len(analysis.simp_theorems)} simp theorems")
    print(f"Found {len(analysis.looping_theorems)} potential looping theorems")
    print(f"Kernel unfoldings: {len(analysis.kernel_unfoldings)}")
    
    for theorem in analysis.get_most_used_theorems():
        print(f"  {theorem.name}: used {theorem.used_count}, tried {theorem.tried_count}, success rate {theorem.success_rate:.2%}")