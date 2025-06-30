#!/usr/bin/env python3
"""
Aggressive optimization script for testing real performance improvements.
This applies a more comprehensive optimization strategy than the default.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

def analyze_simp_rule(rule_text: str) -> dict:
    """Analyze a simp rule to determine optimal priority."""
    # Simple heuristics for priority assignment
    score = 1000  # Default
    
    # High priority for simple, common operations
    if any(pattern in rule_text for pattern in ['+ 0', '0 +', '* 1', '1 *', '* 0', '0 *']):
        score = 2000
    elif any(pattern in rule_text for pattern in ['[]', 'nil', '::[]']):
        score = 1800
    elif 'succ' in rule_text or 'pred' in rule_text:
        score = 1700
    elif 'append' in rule_text or '++' in rule_text:
        score = 1600
    
    # Low priority for complex rules
    elif rule_text.count('(') > 3 or rule_text.count('→') > 2:
        score = 500
    elif len(rule_text) > 200:
        score = 400
    elif rule_text.count('∀') > 1 or rule_text.count('∃') > 0:
        score = 300
    elif 'match' in rule_text or 'if' in rule_text:
        score = 200
    
    # Medium priority for everything else
    else:
        # Count complexity indicators
        complexity = 0
        complexity += rule_text.count('(')
        complexity += rule_text.count('→')
        complexity += rule_text.count('∧')
        complexity += rule_text.count('∨')
        complexity += len(rule_text) // 50
        
        if complexity < 3:
            score = 1500
        elif complexity < 6:
            score = 1000
        else:
            score = 700
    
    return {'priority': score, 'complexity': complexity if 'complexity' in locals() else 0}

def optimize_file(file_path: Path, dry_run: bool = False) -> Tuple[int, List[str]]:
    """Optimize simp rules in a single file."""
    if not file_path.exists():
        return 0, []
    
    content = file_path.read_text()
    lines = content.split('\n')
    
    changes = []
    rules_changed = 0
    new_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for simp attributes
        if '@[simp]' in line:
            # Get the full rule (might span multiple lines)
            rule_text = line
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(('theorem', 'lemma', 'def', '@[')):
                rule_text += ' ' + lines[j].strip()
                j += 1
            
            # Extract theorem/lemma name
            name_match = re.search(r'(theorem|lemma)\s+(\w+)', rule_text)
            if name_match:
                rule_name = name_match.group(2)
                analysis = analyze_simp_rule(rule_text)
                new_priority = analysis['priority']
                
                if new_priority != 1000:  # Only change if not default
                    new_line = line.replace('@[simp]', f'@[simp {new_priority}]')
                    new_lines.append(new_line)
                    rules_changed += 1
                    changes.append(f"{rule_name}: default → {new_priority}")
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
        
        i += 1
    
    # Write back if not dry run
    if rules_changed > 0 and not dry_run:
        file_path.write_text('\n'.join(new_lines))
        print(f"✅ Optimized {file_path.name}: {rules_changed} rules changed")
    
    return rules_changed, changes

def optimize_project(project_path: Path, dry_run: bool = False):
    """Optimize all Lean files in a project."""
    print(f"🧠 Aggressive optimization of {project_path}")
    print("=" * 60)
    
    lean_files = list(project_path.glob("**/*.lean"))
    # Skip lake-packages
    lean_files = [f for f in lean_files if "lake-packages" not in str(f)]
    
    total_changes = 0
    all_changes = []
    
    for lean_file in lean_files:
        changes, change_list = optimize_file(lean_file, dry_run)
        if changes > 0:
            total_changes += changes
            all_changes.extend([(lean_file, c) for c in change_list])
    
    print(f"\n📊 Summary:")
    print(f"  Files analyzed: {len(lean_files)}")
    print(f"  Rules optimized: {total_changes}")
    
    if dry_run:
        print("\n🔍 Proposed changes (dry run):")
        for file, change in all_changes[:20]:  # Show first 20
            print(f"  {file.name}: {change}")
        if len(all_changes) > 20:
            print(f"  ... and {len(all_changes) - 20} more")
    
    print("\n💡 Optimization strategy:")
    print("  - Simple arithmetic (n+0, n*1): Priority 2000")
    print("  - List operations: Priority 1600-1800")
    print("  - Complex patterns: Priority 200-500")
    print("  - Default cases: Priority 1000")
    
    return total_changes

def main():
    if len(sys.argv) < 2:
        print("Usage: python apply_aggressive_optimization.py <project_path> [--dry-run]")
        sys.exit(1)
    
    project_path = Path(sys.argv[1])
    dry_run = "--dry-run" in sys.argv
    
    if not project_path.exists():
        print(f"❌ Error: {project_path} not found")
        sys.exit(1)
    
    changes = optimize_project(project_path, dry_run)
    
    if changes > 0 and not dry_run:
        print("\n✅ Optimization complete! Test with:")
        print(f"  cd {project_path}")
        print("  lake build")

if __name__ == "__main__":
    main()