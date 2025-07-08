"""
Transparency Configuration System

Allows users to control how much optimization information they see
and how the learning process is displayed.
"""

import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TransparencyLevel(Enum):
    """Different levels of transparency"""

    MINIMAL = "minimal"  # Just results
    BASIC = "basic"  # Results + confidence
    DETAILED = "detailed"  # Results + confidence + reasoning
    EXPERT = "expert"  # Full statistical analysis
    DEBUG = "debug"  # Everything including internals


class NotificationStyle(Enum):
    """How to show notifications"""

    SILENT = "silent"  # No notifications
    QUIET = "quiet"  # Minimal text
    NORMAL = "normal"  # Standard notifications
    VERBOSE = "verbose"  # Detailed explanations
    ANIMATED = "animated"  # Rich animations


class LearningVisualization(Enum):
    """Learning curve visualization options"""

    NONE = "none"  # No visualization
    TEXT = "text"  # Text-based progress
    SIMPLE = "simple"  # Basic charts
    RICH = "rich"  # Full interactive charts
    REALTIME = "realtime"  # Live updating dashboard


@dataclass
class TransparencyPreferences:
    """User preferences for transparency and explanations"""

    # Core transparency settings
    transparency_level: TransparencyLevel = TransparencyLevel.DETAILED
    show_confidence_intervals: bool = True
    show_strategy_reasoning: bool = True
    show_learning_progress: bool = True
    show_network_contribution: bool = True

    # Notification preferences
    notification_style: NotificationStyle = NotificationStyle.NORMAL
    show_success_notifications: bool = True
    show_failure_explanations: bool = True
    show_optimization_tips: bool = True

    # Visualization preferences
    learning_visualization: LearningVisualization = LearningVisualization.RICH
    show_context_breakdown: bool = True
    show_strategy_heatmap: bool = True
    show_regret_curves: bool = False

    # Advanced features
    enable_interactive_explanations: bool = True
    show_mathematical_details: bool = False
    export_learning_data: bool = False

    # Dashboard preferences
    auto_refresh_interval: int = 5  # seconds
    compact_mode: bool = False
    dark_theme: bool = True

    # Privacy and contribution
    contribute_to_global_model: bool = False
    share_anonymized_patterns: bool = False
    enable_federated_learning: bool = False


class TransparencyConfig:
    """Manages transparency configuration"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".simpulse" / "transparency.json"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._preferences = self._load_preferences()

    def _load_preferences(self) -> TransparencyPreferences:
        """Load preferences from config file"""

        if not self.config_path.exists():
            # Create default config
            prefs = TransparencyPreferences()
            self._save_preferences(prefs)
            return prefs

        try:
            with open(self.config_path) as f:
                data = json.load(f)

            # Convert enum strings back to enums
            for field in ["transparency_level", "notification_style", "learning_visualization"]:
                if field in data:
                    enum_class = {
                        "transparency_level": TransparencyLevel,
                        "notification_style": NotificationStyle,
                        "learning_visualization": LearningVisualization,
                    }[field]
                    data[field] = enum_class(data[field])

            # Create preferences object
            return TransparencyPreferences(**data)

        except Exception as e:
            logger.warning(f"Failed to load transparency config: {e}")
            return TransparencyPreferences()

    def _save_preferences(self, preferences: TransparencyPreferences):
        """Save preferences to config file"""

        # Convert to dict and handle enums
        data = asdict(preferences)
        for key, value in data.items():
            if isinstance(value, Enum):
                data[key] = value.value

        try:
            with open(self.config_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save transparency config: {e}")

    @property
    def preferences(self) -> TransparencyPreferences:
        """Get current preferences"""
        return self._preferences

    def update_preferences(self, **kwargs):
        """Update specific preferences"""

        for key, value in kwargs.items():
            if hasattr(self._preferences, key):
                setattr(self._preferences, key, value)
            else:
                logger.warning(f"Unknown preference: {key}")

        self._save_preferences(self._preferences)

    def get_explanation_template(
        self, context: str, strategy: str, confidence: float
    ) -> Dict[str, Any]:
        """Get explanation template based on transparency level"""

        level = self._preferences.transparency_level

        if level == TransparencyLevel.MINIMAL:
            return {
                "show_strategy": True,
                "show_confidence": False,
                "show_reasoning": False,
                "show_factors": False,
                "show_statistics": False,
            }
        elif level == TransparencyLevel.BASIC:
            return {
                "show_strategy": True,
                "show_confidence": True,
                "show_reasoning": False,
                "show_factors": False,
                "show_statistics": False,
            }
        elif level == TransparencyLevel.DETAILED:
            return {
                "show_strategy": True,
                "show_confidence": True,
                "show_reasoning": True,
                "show_factors": True,
                "show_statistics": False,
            }
        elif level == TransparencyLevel.EXPERT:
            return {
                "show_strategy": True,
                "show_confidence": True,
                "show_reasoning": True,
                "show_factors": True,
                "show_statistics": True,
                "show_regret": True,
                "show_convergence": True,
            }
        else:  # DEBUG
            return {
                "show_strategy": True,
                "show_confidence": True,
                "show_reasoning": True,
                "show_factors": True,
                "show_statistics": True,
                "show_regret": True,
                "show_convergence": True,
                "show_bandit_state": True,
                "show_raw_features": True,
            }

    def should_show_notification(self, notification_type: str) -> bool:
        """Check if a notification type should be shown"""

        style = self._preferences.notification_style

        if style == NotificationStyle.SILENT:
            return False
        elif style == NotificationStyle.QUIET:
            return notification_type in ["critical_failure", "major_success"]
        elif style == NotificationStyle.NORMAL:
            return notification_type in ["success", "failure", "tip", "progress"]
        elif style == NotificationStyle.VERBOSE:
            return True  # Show all notifications
        else:  # ANIMATED
            return True

    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration"""

        return {
            "auto_refresh": self._preferences.auto_refresh_interval,
            "compact_mode": self._preferences.compact_mode,
            "dark_theme": self._preferences.dark_theme,
            "show_learning_curves": self._preferences.learning_visualization
            != LearningVisualization.NONE,
            "show_context_breakdown": self._preferences.show_context_breakdown,
            "show_strategy_heatmap": self._preferences.show_strategy_heatmap,
            "show_regret_curves": self._preferences.show_regret_curves,
            "enable_interactions": self._preferences.enable_interactive_explanations,
            "show_math": self._preferences.show_mathematical_details,
        }

    def get_privacy_settings(self) -> Dict[str, bool]:
        """Get privacy and contribution settings"""

        return {
            "contribute_to_global": self._preferences.contribute_to_global_model,
            "share_patterns": self._preferences.share_anonymized_patterns,
            "federated_learning": self._preferences.enable_federated_learning,
            "export_data": self._preferences.export_learning_data,
        }

    def create_user_profile(self) -> str:
        """Create a user profile description for personalized explanations"""

        prefs = self._preferences

        if prefs.transparency_level == TransparencyLevel.MINIMAL:
            return "prefers_minimal_info"
        elif prefs.transparency_level == TransparencyLevel.EXPERT:
            return "technical_expert"
        elif prefs.show_mathematical_details:
            return "mathematical_user"
        elif prefs.learning_visualization == LearningVisualization.REALTIME:
            return "data_enthusiast"
        else:
            return "standard_user"

    def generate_personalized_explanation(self, optimization_result: Dict[str, Any]) -> str:
        """Generate explanation text based on user preferences"""

        profile = self.create_user_profile()
        template = self.get_explanation_template(
            optimization_result.get("context", ""),
            optimization_result.get("strategy", ""),
            optimization_result.get("confidence", 0.0),
        )

        explanation = []

        # Strategy selection
        if template.get("show_strategy"):
            strategy = optimization_result.get("strategy", "unknown")
            explanation.append(f"Selected {strategy} optimization")

        # Confidence
        if template.get("show_confidence"):
            confidence = optimization_result.get("confidence", 0.0)
            conf_text = f"with {confidence:.0%} confidence"

            if profile == "technical_expert":
                # Add confidence interval
                ci_lower = max(0, confidence - 0.1)
                ci_upper = min(1, confidence + 0.1)
                conf_text += f" (CI: {ci_lower:.0%}-{ci_upper:.0%})"

            explanation.append(conf_text)

        # Reasoning
        if template.get("show_reasoning"):
            context = optimization_result.get("context", "")
            if context:
                if profile == "mathematical_user":
                    explanation.append(f"based on Bayesian inference over {context} patterns")
                else:
                    explanation.append(f"because your file contains mostly {context} patterns")

        # Performance prediction
        expected_speedup = optimization_result.get("expected_speedup", 1.0)
        if expected_speedup > 1.1:
            if profile == "prefers_minimal_info":
                explanation.append(f"→ {expected_speedup:.1f}× faster")
            else:
                explanation.append(f"Expected speedup: {expected_speedup:.1f}×")

        # Statistics (for experts)
        if template.get("show_statistics"):
            trials = optimization_result.get("trials", 0)
            if trials > 0:
                explanation.append(f"(based on {trials} similar files)")

        return " ".join(explanation)

    def setup_interactive_tutorial(self) -> Dict[str, Any]:
        """Set up interactive tutorial for new users"""

        return {
            "show_tutorial": self._preferences.transparency_level != TransparencyLevel.MINIMAL,
            "tutorial_steps": [
                {
                    "title": "Welcome to Transparent Optimization",
                    "content": "Simpulse shows you exactly how it learns to optimize your code.",
                    "interactive": self._preferences.enable_interactive_explanations,
                },
                {
                    "title": "Understanding Context",
                    "content": "We analyze your file patterns to choose the best optimization strategy.",
                    "show_example": True,
                },
                {
                    "title": "Confidence & Learning",
                    "content": "Higher confidence means better predictions based on similar files.",
                    "show_chart": self._preferences.learning_visualization
                    != LearningVisualization.NONE,
                },
                {
                    "title": "Network Benefits",
                    "content": "Contributing anonymously helps everyone optimize faster.",
                    "show_privacy": True,
                },
            ],
        }


class ExplanationGenerator:
    """Generates context-aware explanations"""

    def __init__(self, config: TransparencyConfig):
        self.config = config

    def explain_strategy_choice(
        self, context: Dict[str, float], strategy: str, confidence: float, factors: Dict[str, float]
    ) -> str:
        """Generate strategy choice explanation"""

        prefs = self.config.preferences
        profile = self.config.create_user_profile()

        # Get dominant context
        dominant = max(context.items(), key=lambda x: x[1])

        explanations = {
            "prefers_minimal_info": f"{strategy} → {confidence:.0%}",
            "standard_user": f"Using {strategy} for {dominant[0]} files ({confidence:.0%} confident)",
            "technical_expert": f"Strategy: {strategy} | Confidence: {confidence:.2f} ± 0.1 | Context: {dominant[0]} ({dominant[1]:.2f})",
            "mathematical_user": f"argmax P(speedup|{strategy}, context) = {confidence:.3f}",
        }

        base_explanation = explanations.get(profile, explanations["standard_user"])

        # Add reasoning if enabled
        if prefs.show_strategy_reasoning:
            if strategy.endswith("_pure") and strategy.startswith(dominant[0][:4]):
                base_explanation += " (specialized strategy matches context)"
            elif strategy == "weighted_hybrid":
                base_explanation += " (hybrid strategy for mixed patterns)"
            elif factors.get("high_confidence", 0) > 0.8:
                base_explanation += " (high confidence from similar files)"

        return base_explanation

    def explain_learning_progress(self, progress_data: Dict[str, Any]) -> str:
        """Explain current learning progress"""

        profile = self.config.create_user_profile()

        success_rate = progress_data.get("success_rate", 0)
        trials = progress_data.get("trials", 0)
        learning_rate = progress_data.get("learning_rate", 0)

        if profile == "prefers_minimal_info":
            return f"{success_rate:.0%} success ({trials} files)"

        elif profile == "mathematical_user":
            regret = trials * (1 - success_rate)
            return f"Cumulative regret: {regret:.1f} | Learning rate: {learning_rate:.3f}"

        elif profile == "technical_expert":
            ci_lower = max(
                0, success_rate - 1.96 * (success_rate * (1 - success_rate) / trials) ** 0.5
            )
            ci_upper = min(
                1, success_rate + 1.96 * (success_rate * (1 - success_rate) / trials) ** 0.5
            )
            return (
                f"Success: {success_rate:.1%} (95% CI: {ci_lower:.1%}-{ci_upper:.1%}) | n={trials}"
            )

        else:  # standard_user
            if trials < 10:
                return f"Learning... {success_rate:.0%} success in {trials} attempts"
            elif success_rate > 0.8:
                return f"Optimizing well! {success_rate:.0%} success rate"
            elif learning_rate > 0.05:
                return f"Rapidly improving: {success_rate:.0%} success, trending up"
            else:
                return f"Steady performance: {success_rate:.0%} success rate"


def create_transparency_cli():
    """Create CLI for configuring transparency preferences"""

    import click
    from rich.console import Console
    from rich.prompt import Confirm, Prompt
    from rich.table import Table

    console = Console()
    config = TransparencyConfig()

    @click.group()
    def transparency():
        """Configure Simpulse transparency and explanations"""

    @transparency.command()
    def show():
        """Show current transparency settings"""

        prefs = config.preferences

        # Create settings table
        table = Table(title="Transparency Settings")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Transparency Level", prefs.transparency_level.value)
        table.add_row("Notification Style", prefs.notification_style.value)
        table.add_row("Learning Visualization", prefs.learning_visualization.value)
        table.add_row("Show Confidence", "✅" if prefs.show_confidence_intervals else "❌")
        table.add_row("Strategy Reasoning", "✅" if prefs.show_strategy_reasoning else "❌")
        table.add_row("Global Contribution", "✅" if prefs.contribute_to_global_model else "❌")

        console.print(table)

    @transparency.command()
    def setup():
        """Interactive setup wizard"""

        console.print("[bold]🎯 Simpulse Transparency Setup[/bold]")
        console.print("Let's configure how much optimization detail you want to see.\n")

        # Transparency level
        console.print("How much detail do you want about optimization decisions?")
        console.print("1. Minimal - Just show results")
        console.print("2. Basic - Results + confidence")
        console.print("3. Detailed - Results + confidence + reasoning")
        console.print("4. Expert - Full statistical analysis")

        level_choice = Prompt.ask("Choose level", choices=["1", "2", "3", "4"], default="3")
        level_map = {
            "1": TransparencyLevel.MINIMAL,
            "2": TransparencyLevel.BASIC,
            "3": TransparencyLevel.DETAILED,
            "4": TransparencyLevel.EXPERT,
        }

        # Visualization preference
        console.print("\nHow do you want to see learning progress?")
        console.print("1. None - No visualization")
        console.print("2. Text - Text-based progress")
        console.print("3. Simple - Basic charts")
        console.print("4. Rich - Interactive dashboard")

        viz_choice = Prompt.ask("Choose visualization", choices=["1", "2", "3", "4"], default="4")
        viz_map = {
            "1": LearningVisualization.NONE,
            "2": LearningVisualization.TEXT,
            "3": LearningVisualization.SIMPLE,
            "4": LearningVisualization.RICH,
        }

        # Contribution preference
        contribute = Confirm.ask(
            "\nContribute anonymously to global optimization model for 10× faster learning?"
        )

        # Update preferences
        config.update_preferences(
            transparency_level=level_map[level_choice],
            learning_visualization=viz_map[viz_choice],
            contribute_to_global_model=contribute,
        )

        console.print("\n[green]✅ Configuration saved![/green]")
        console.print("Run 'simpulse optimize <file>' to see your new settings in action.")

    return transparency


if __name__ == "__main__":
    cli = create_transparency_cli()
    cli()
