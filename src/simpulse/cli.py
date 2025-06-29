#!/usr/bin/env python3
"""Minimal Simpulse CLI - just the essentials."""

import asyncio
import sys
from pathlib import Path

from .evolution.evolution_engine import SimpleEvolutionEngine


async def optimize_command(file_path: str):
    """Optimize a single Lean file."""
    
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File {file_path} not found")
        return 1
        
    print(f"Optimizing {path.name}...")
    
    engine = SimpleEvolutionEngine()
    result = await engine.optimize_file(path)
    
    if result.improved:
        print(f"✅ Success! {result.improvement_percent:.1f}% improvement")
        print(f"   Baseline: {result.baseline_time:.2f}ms")
        print(f"   Optimized: {result.optimized_time:.2f}ms")
        print(f"   Best mutation: {result.best_mutation}")
    else:
        print("❌ No improvement found")
        print("   Try a file with more simp rules")
        
    return 0 if result.improved else 1


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: simpulse optimize <file.lean>")
        print("       simpulse --help")
        return 1
        
    command = sys.argv[1]
    
    if command == "--help" or command == "-h":
        print("Simpulse - Lean 4 simp optimizer")
        print()
        print("Commands:")
        print("  optimize <file>  Optimize simp rules in a Lean file")
        print("  --help          Show this help")
        return 0
        
    elif command == "optimize" and len(sys.argv) > 2:
        return asyncio.run(optimize_command(sys.argv[2]))
        
    else:
        print(f"Unknown command: {command}")
        print("Run 'simpulse --help' for usage")
        return 1


if __name__ == "__main__":
    sys.exit(main())
