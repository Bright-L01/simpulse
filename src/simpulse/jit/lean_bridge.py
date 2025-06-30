"""
Bridge between Python runtime adapter and Lean 4 JIT profiler.

Provides integration mechanisms for transparent activation.
"""

import subprocess
from pathlib import Path
from typing import Optional

from .runtime_adapter import AdapterConfig, RuntimeAdapter


class LeanJITBridge:
    """Bridge for integrating JIT profiler with Lean projects."""

    def __init__(self, project_path: str, config: Optional[AdapterConfig] = None):
        self.project_path = Path(project_path)
        self.config = config or AdapterConfig()
        self.adapter = RuntimeAdapter(config)

        # Paths for Lean integration
        self.lean_ext_path = self.project_path / ".lake" / "packages" / "simpulse-jit"
        self.priorities_path = self.project_path / "simp_priorities.json"
        self.stats_path = self.project_path / "simp_stats.json"

    def setup_lean_extension(self) -> bool:
        """Set up Simpulse JIT as a Lean extension."""
        try:
            # Check if lake is available
            result = subprocess.run(
                ["lake", "--version"], capture_output=True, text=True
            )
            if result.returncode != 0:
                print("Error: lake not found. Please install Lean 4.")
                return False

            # Add Simpulse JIT to lakefile
            lakefile_path = self.project_path / "lakefile.lean"
            if lakefile_path.exists():
                content = lakefile_path.read_text()

                # Check if already added
                if "simpulse-jit" not in content:
                    # Add dependency
                    dependency = """
require «simpulse-jit» from git
  "https://github.com/bright-L01/simpulse" / "lean4" / "SimpulseJIT"
"""

                    # Insert after package declaration
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        if line.startswith("package"):
                            lines.insert(i + 1, dependency)
                            break

                    lakefile_path.write_text("\n".join(lines))
                    print("Added Simpulse JIT dependency to lakefile.lean")

            # Update dependencies
            subprocess.run(["lake", "update"], cwd=self.project_path, check=True)

            return True

        except Exception as e:
            print(f"Error setting up Lean extension: {e}")
            return False

    def enable_jit_profiling(self) -> None:
        """Enable JIT profiling for the project."""
        # Set environment variables
        env_vars = {
            "SIMPULSE_JIT": "1",
            "SIMPULSE_JIT_SAVE": str(self.priorities_path),
            "SIMPULSE_JIT_LOG": str(self.stats_path),
            "SIMPULSE_JIT_INTERVAL": str(self.config.adaptation_interval),
        }

        # Create .env file for the project
        env_file = self.project_path / ".env"
        with open(env_file, "w") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")

        print(f"JIT profiling enabled for {self.project_path}")
        print("Environment variables set in .env file")

    def inject_profiling_import(self) -> None:
        """Inject profiling import into Lean files."""
        # Find all Lean files
        lean_files = list(self.project_path.rglob("*.lean"))

        import_line = "import SimpulseJIT.Integration\n"

        modified = 0
        for lean_file in lean_files:
            content = lean_file.read_text()

            # Skip if already imported
            if "SimpulseJIT" in content:
                continue

            # Add import after other imports
            lines = content.split("\n")
            import_idx = 0

            for i, line in enumerate(lines):
                if line.strip().startswith("import"):
                    import_idx = i + 1
                elif import_idx > 0 and not line.strip().startswith("import"):
                    break

            # Insert import
            lines.insert(import_idx, import_line.strip())

            # Enable JIT for this file
            if "namespace" in content:
                # Add after namespace declaration
                for i, line in enumerate(lines):
                    if line.strip().startswith("namespace"):
                        lines.insert(i + 1, "enable_jit_profiling")
                        break
            else:
                # Add at the beginning after imports
                lines.insert(import_idx + 1, "enable_jit_profiling")

            lean_file.write_text("\n".join(lines))
            modified += 1

        print(f"Modified {modified} Lean files with JIT profiling")

    def apply_optimized_priorities(self) -> None:
        """Apply optimized priorities to Lean project."""
        # Load current priorities
        priorities = self.adapter.load_priorities()

        if not priorities:
            print("No optimized priorities found")
            return

        # Create priority override file
        override_content = [
            "/-",
            "  Optimized simp priorities from Simpulse JIT",
            "-/",
            "",
            "namespace SimpulseJIT.Overrides",
            "",
        ]

        for rule_name, priority in sorted(priorities.items()):
            # Generate priority override
            override_content.append(f"attribute [simp {priority}] {rule_name}")

        override_content.append("")
        override_content.append("end SimpulseJIT.Overrides")

        # Save override file
        override_file = self.project_path / "SimpulseJITPriorities.lean"
        override_file.write_text("\n".join(override_content))

        print(f"Applied {len(priorities)} optimized priorities")
        print(f"Priorities saved to: {override_file}")

    def monitor_and_optimize(self) -> None:
        """Monitor Lean builds and optimize priorities."""
        print(f"Monitoring Lean project at {self.project_path}")
        print("Press Ctrl+C to stop")

        try:
            # Watch for statistics updates
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer

            class StatsHandler(FileSystemEventHandler):
                def __init__(self, bridge):
                    self.bridge = bridge

                def on_modified(self, event):
                    if event.src_path == str(self.bridge.stats_path):
                        # Reload statistics
                        self.bridge.adapter.load_statistics()

                        # Check if optimization needed
                        if (
                            self.bridge.adapter.call_count
                            % self.bridge.config.adaptation_interval
                            == 0
                        ):
                            priorities = self.bridge.adapter.optimize_priorities()
                            if priorities:
                                self.bridge.apply_optimized_priorities()
                                print(f"Optimized {len(priorities)} priorities")

            observer = Observer()
            handler = StatsHandler(self)
            observer.schedule(handler, str(self.project_path), recursive=False)
            observer.start()

            # Keep running
            import time

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            observer.stop()
            print("\nMonitoring stopped")

        observer.join()

    def generate_report(self) -> str:
        """Generate performance report."""
        # Get statistics summary
        summary = self.adapter.get_statistics_summary()

        # Add project-specific information
        report = [
            f"=== Simpulse JIT Report for {self.project_path.name} ===",
            "",
            summary,
            "",
            "Optimization Settings:",
            f"  Adaptation interval: {self.config.adaptation_interval} calls",
            f"  Decay factor: {self.config.decay_factor}",
            f"  Min samples: {self.config.min_samples}",
            f"  Priority range: {self.config.priority_range}",
            "",
            "Files:",
            f"  Statistics: {self.stats_path}",
            f"  Priorities: {self.priorities_path}",
        ]

        return "\n".join(report)


def setup_jit_for_project(project_path: str) -> None:
    """Quick setup for JIT profiling in a Lean project."""
    bridge = LeanJITBridge(project_path)

    print("Setting up Simpulse JIT profiling...")

    # 1. Set up Lean extension
    if not bridge.setup_lean_extension():
        print("Failed to set up Lean extension")
        return

    # 2. Enable profiling
    bridge.enable_jit_profiling()

    # 3. Inject imports
    bridge.inject_profiling_import()

    # 4. Apply any existing priorities
    bridge.apply_optimized_priorities()

    print("\nSetup complete!")
    print(f"Run 'lake build' in {project_path} to start profiling")
    print(
        f"Use 'python -m simpulse.jit monitor {project_path}' to monitor and optimize"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m simpulse.jit.lean_bridge <project_path>")
        sys.exit(1)

    project_path = sys.argv[1]

    if len(sys.argv) > 2 and sys.argv[2] == "monitor":
        # Monitor mode
        bridge = LeanJITBridge(project_path)
        bridge.monitor_and_optimize()
    else:
        # Setup mode
        setup_jit_for_project(project_path)
