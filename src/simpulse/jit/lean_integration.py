#!/usr/bin/env python3
"""
Integration layer between JIT optimizer and Lean 4.

Provides hooks for Lean to communicate with the Python optimizer.
"""

import json
import os
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .dynamic_optimizer import DynamicSimpOptimizer, OptimizationContext


@dataclass
class SimpAttemptMessage:
    """Message from Lean about a simp attempt."""

    rule_name: str
    module_name: str
    goal_type: str
    proof_depth: int
    previous_tactics: list[str]
    timestamp: float


@dataclass
class SimpResultMessage:
    """Result of a simp attempt."""

    rule_name: str
    success: bool
    duration: float
    timestamp: float


@dataclass
class PriorityQueryMessage:
    """Query for rule priorities."""

    rules: list[str]
    context_module: str
    context_goal_type: str


@dataclass
class PriorityResponseMessage:
    """Response with rule priorities."""

    priorities: dict[str, int]


class LeanJITServer:
    """Server for JIT optimization communication with Lean."""

    def __init__(
        self,
        socket_path: str = "/tmp/simpulse_jit.sock",
        config_path: str = "simpulse_jit_config.json",
    ):
        """Initialize JIT server."""
        self.socket_path = socket_path
        self.config_path = Path(config_path)
        self.optimizer = DynamicSimpOptimizer()
        self.running = False
        self.server_thread = None

        # Load existing configuration if available
        self._load_config()

        # Performance metrics
        self.messages_processed = 0
        self.last_save_time = time.time()

    def _load_config(self):
        """Load existing optimization data."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    config = json.load(f)

                # Restore priorities
                if "priorities" in config:
                    self.optimizer.priority_cache = config["priorities"]

                print(f"Loaded optimization config from {self.config_path}")
            except Exception as e:
                print(f"Error loading config: {e}")

    def start(self):
        """Start the JIT server."""
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        print(f"JIT server started on {self.socket_path}")

    def stop(self):
        """Stop the JIT server."""
        self.running = False
        if self.server_thread:
            self.server_thread.join()
        self._save_config()
        print("JIT server stopped")

    def _run_server(self):
        """Main server loop using Unix domain sockets."""
        # Remove existing socket file
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # Create Unix domain socket
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.socket_path)
        server.listen(5)
        server.settimeout(1.0)  # Allow periodic checks

        print(f"Listening on {self.socket_path}")

        while self.running:
            try:
                # Accept connection
                conn, _ = server.accept()

                # Handle in separate thread
                handler = threading.Thread(target=self._handle_connection, args=(conn,))
                handler.daemon = True
                handler.start()

            except TimeoutError:
                # Periodic maintenance
                self._periodic_maintenance()
                continue
            except Exception as e:
                print(f"Server error: {e}")

        server.close()
        os.unlink(self.socket_path)

    def _handle_connection(self, conn: socket.socket):
        """Handle a client connection."""
        try:
            # Receive message
            data = conn.recv(4096)
            if not data:
                return

            message = json.loads(data.decode("utf-8"))

            # Process message
            response = self._process_message(message)

            # Send response
            if response:
                conn.send(json.dumps(response).encode("utf-8"))

            self.messages_processed += 1

        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            conn.close()

    def _process_message(self, message: dict) -> dict | None:
        """Process incoming message from Lean."""
        msg_type = message.get("type")

        if msg_type == "simp_attempt":
            # Record simp attempt
            context = OptimizationContext(
                module_name=message["module_name"],
                goal_type=message["goal_type"],
                proof_depth=message["proof_depth"],
                previous_tactics=message.get("previous_tactics", []),
            )

            # Start tracking (Lean will send result later)
            return {"status": "tracking"}

        elif msg_type == "simp_result":
            # Record result
            self.optimizer.record_attempt(
                rule_name=message["rule_name"],
                success=message["success"],
                duration=message["duration"],
                context=None,  # Context was in attempt message
            )
            return {"status": "recorded"}

        elif msg_type == "priority_query":
            # Return priorities for rules
            priorities = {}
            context = OptimizationContext(
                module_name=message.get("context_module", ""),
                goal_type=message.get("context_goal_type", ""),
                proof_depth=0,
                previous_tactics=[],
            )

            for rule in message["rules"]:
                priorities[rule] = self.optimizer.get_priority(rule, context)

            return {"type": "priority_response", "priorities": priorities}

        elif msg_type == "get_stats":
            # Return current statistics
            stats = {
                "total_attempts": self.optimizer.global_attempts,
                "unique_rules": len(self.optimizer.rule_stats),
                "avg_improvement": (
                    sum(self.optimizer.improvements) / len(self.optimizer.improvements)
                    if self.optimizer.improvements
                    else 0
                ),
                "messages_processed": self.messages_processed,
            }
            return {"type": "stats", "data": stats}

        return None

    def _periodic_maintenance(self):
        """Perform periodic maintenance tasks."""
        current_time = time.time()

        # Save configuration every 5 minutes
        if current_time - self.last_save_time > 300:
            self._save_config()
            self.last_save_time = current_time

    def _save_config(self):
        """Save current optimization state."""
        self.optimizer.compile_optimized_simp(self.config_path)


class LeanJITClient:
    """Client for Lean to communicate with JIT optimizer."""

    def __init__(self, socket_path: str = "/tmp/simpulse_jit.sock"):
        """Initialize JIT client."""
        self.socket_path = socket_path

    def notify_attempt(
        self,
        rule_name: str,
        module_name: str,
        goal_type: str,
        proof_depth: int = 0,
        previous_tactics: list[str] = None,
    ) -> bool:
        """Notify JIT optimizer of simp attempt."""
        message = {
            "type": "simp_attempt",
            "rule_name": rule_name,
            "module_name": module_name,
            "goal_type": goal_type,
            "proof_depth": proof_depth,
            "previous_tactics": previous_tactics or [],
        }

        return self._send_message(message)

    def notify_result(self, rule_name: str, success: bool, duration: float) -> bool:
        """Notify JIT optimizer of attempt result."""
        message = {
            "type": "simp_result",
            "rule_name": rule_name,
            "success": success,
            "duration": duration,
        }

        return self._send_message(message)

    def get_priorities(
        self, rules: list[str], context_module: str = "", context_goal_type: str = ""
    ) -> dict[str, int]:
        """Get current priorities for rules."""
        message = {
            "type": "priority_query",
            "rules": rules,
            "context_module": context_module,
            "context_goal_type": context_goal_type,
        }

        response = self._send_message(message, expect_response=True)

        if response and response.get("type") == "priority_response":
            return response["priorities"]

        # Default priorities if server unavailable
        return {rule: 1000 for rule in rules}

    def get_stats(self) -> dict[str, any]:
        """Get current optimization statistics."""
        message = {"type": "get_stats"}
        response = self._send_message(message, expect_response=True)

        if response and response.get("type") == "stats":
            return response["data"]

        return {}

    def _send_message(self, message: dict, expect_response: bool = False) -> any:
        """Send message to JIT server."""
        try:
            # Connect to server
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(1.0)
            client.connect(self.socket_path)

            # Send message
            client.send(json.dumps(message).encode("utf-8"))

            # Get response if expected
            if expect_response:
                data = client.recv(4096)
                response = json.loads(data.decode("utf-8"))
                client.close()
                return response

            client.close()
            return True

        except Exception:
            # Server might not be running, fail gracefully
            return None if expect_response else False


def create_lean_bindings(output_path: Path):
    """Generate Lean 4 bindings for JIT integration."""
    lean_code = """-- Simpulse JIT Integration for Lean 4
import Lean

namespace Simpulse.JIT

/-- Foreign function interface for JIT communication -/
@[extern "simpulse_jit_notify_attempt"]
opaque notifyAttempt (ruleName : String) (moduleName : String) 
                    (goalType : String) (proofDepth : Nat) : IO Unit

@[extern "simpulse_jit_notify_result"]
opaque notifyResult (ruleName : String) (success : Bool) 
                   (duration : Float) : IO Unit

@[extern "simpulse_jit_get_priority"]
opaque getPriority (ruleName : String) (moduleName : String) 
                  (goalType : String) : IO Nat

/-- Hook into simp tactic execution -/
def instrumentedSimpAttempt (ruleName : Name) (goal : MVarId) 
                           (action : MetaM (Option Expr)) : MetaM (Option Expr) := do
  let startTime ← IO.monoMsNow
  
  -- Get context
  let ctx ← getMainModule
  let goalType := "unknown"  -- Would analyze goal structure
  let depth ← getProofDepth
  
  -- Notify JIT
  IO.ofExcept $ notifyAttempt ruleName.toString ctx.toString goalType depth
  
  -- Execute simp
  let result ← action
  
  -- Calculate duration
  let endTime ← IO.monoMsNow
  let duration := (endTime - startTime).toFloat / 1000.0
  
  -- Notify result
  IO.ofExcept $ notifyResult ruleName.toString result.isSome duration
  
  return result

/-- Apply JIT-optimized priorities -/
def applyJITPriorities : MetaM Unit := do
  let rules ← getSimpRules
  let ctx ← getMainModule
  
  for rule in rules do
    let priority ← IO.ofExcept $ getPriority rule.toString ctx.toString "general"
    setSimpPriority rule priority

-- Initialize JIT optimization
initialize do
  -- Start JIT server if configured
  if (← IO.getEnv "SIMPULSE_JIT_ENABLED").isSome then
    applyJITPriorities

end Simpulse.JIT
"""

    output_path.write_text(lean_code)
    print(f"Lean bindings saved to {output_path}")


def demo_jit_optimization():
    """Demonstrate JIT optimization in action."""
    print("=== Simpulse JIT Optimization Demo ===\n")

    # Start server
    server = LeanJITServer()
    server.start()

    # Wait for server to start
    time.sleep(0.5)

    # Create client
    client = LeanJITClient()

    print("Simulating Lean proof execution...\n")

    # Simulate proof with multiple simp calls
    rules = [
        "List.append_nil",
        "Nat.add_zero",
        "List.map_append",
        "ComplexRule1",
        "ComplexRule2",
    ]

    for i in range(20):
        print(f"Proof {i+1}:")

        # Get current priorities
        priorities = client.get_priorities(rules, "TestModule", "arithmetic")

        # Sort by priority
        sorted_rules = sorted(rules, key=lambda r: priorities.get(r, 1000))
        print(f"  Trying rules in order: {sorted_rules}")

        # Simulate trying rules
        for rule in sorted_rules:
            client.notify_attempt(rule, "TestModule", "arithmetic", i % 3)

            # Simulate execution
            exec_time = 0.001 if "Complex" not in rule else 0.01
            time.sleep(exec_time)

            # Simulate success (simple rules more likely)
            success = (i % 3 == 0) if "Complex" in rule else True

            client.notify_result(rule, success, exec_time)

            if success:
                print(f"  ✓ Succeeded with {rule}")
                break

        print()

        # Show stats periodically
        if (i + 1) % 5 == 0:
            stats = client.get_stats()
            print(f"Current stats: {stats}\n")

    # Final statistics
    print("\nFinal Optimization Results:")
    final_stats = client.get_stats()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")

    # Stop server
    server.stop()


if __name__ == "__main__":
    # Run demo
    demo_jit_optimization()

    # Generate Lean bindings
    create_lean_bindings(Path("SimpulseJIT.lean"))
