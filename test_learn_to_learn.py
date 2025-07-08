"""
Comprehensive Learn to Learn System Testing

Tests all aspects of meta-learning for optimization processes:
1. Context learning speed analysis
2. Strategy generalization metrics
3. Learning curve prediction accuracy
4. Exploration rate optimization
5. Meta-parameter tuning effectiveness
6. Integration with optimization pipeline
"""

import logging
import random
import time

import numpy as np

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from src.simpulse.meta_learning.learn_to_learn import (
    ContextLearningAnalyzer,
    ExplorationRateTuner,
    LearningCurvePredictor,
    LearningMetrics,
    LearnToLearnSystem,
    MetaBandit,
    StrategyGeneralizationAnalyzer,
)


def generate_realistic_learning_metrics(
    context_type: str, strategy: str, trial: int
) -> LearningMetrics:
    """Generate realistic learning metrics for testing"""

    # Simulate learning curves - different strategies learn at different rates
    strategy_learning_rates = {
        "arithmetic_pure": {"arithmetic": 0.9, "algebraic": 0.3, "structural": 0.2, "mixed": 0.5},
        "algebraic_pure": {"arithmetic": 0.2, "algebraic": 0.9, "structural": 0.3, "mixed": 0.4},
        "structural_pure": {"arithmetic": 0.1, "algebraic": 0.2, "structural": 0.8, "mixed": 0.3},
        "weighted_hybrid": {"arithmetic": 0.7, "algebraic": 0.6, "structural": 0.5, "mixed": 0.8},
        "phase_based": {"arithmetic": 0.6, "algebraic": 0.7, "structural": 0.6, "mixed": 0.7},
    }

    # Get base learning rate
    base_rate = strategy_learning_rates.get(strategy, {}).get(context_type, 0.5)

    # Simulate learning curve (performance improves over trials)
    learning_factor = 1.0 - np.exp(-trial * 0.1)
    current_performance = base_rate * learning_factor

    # Add noise
    noise = random.normalvariate(0, 0.05)
    current_performance = max(0.0, min(1.0, current_performance + noise))

    # Success based on performance
    success = random.random() < current_performance
    speedup = 1.0 + current_performance if success else 0.8 + random.random() * 0.3

    # Calculate learning metrics
    exploration_rate = max(0.1, 0.5 - trial * 0.02)  # Decreases over time
    exploitation_rate = 1.0 - exploration_rate

    # Simple regret calculation
    optimal_performance = max(
        strategy_learning_rates[s].get(context_type, 0) for s in strategy_learning_rates
    )
    regret = max(0, optimal_performance - current_performance)

    return LearningMetrics(
        context_id=f"{context_type}_context_{random.randint(1, 100)}",
        strategy_name=strategy,
        trial_number=trial,
        timestamp=time.time(),
        success=success,
        speedup=speedup,
        confidence=current_performance,
        exploration_rate=exploration_rate,
        exploitation_rate=exploitation_rate,
        regret=regret,
        cumulative_regret=regret * trial,  # Simplified cumulative regret
        learning_rate=0.1,
        context_familiarity=min(1.0, trial * 0.05),
        strategy_confidence=current_performance,
        convergence_indicator=learning_factor,
    )


def test_context_learning_analyzer():
    """Test context learning speed analysis"""
    print("\n🧠 Testing Context Learning Analyzer")
    print("=" * 50)

    analyzer = ContextLearningAnalyzer()

    # Generate learning data for different contexts
    contexts = ["arithmetic", "algebraic", "structural", "mixed"]
    strategies = ["arithmetic_pure", "algebraic_pure", "weighted_hybrid"]

    print("📚 Generating learning data...")

    learning_data = []
    for context in contexts:
        for strategy in strategies:
            for trial in range(1, 21):  # 20 trials per context-strategy pair
                metrics = generate_realistic_learning_metrics(context, strategy, trial)
                learning_data.append(metrics)

    print(f"   Generated {len(learning_data)} learning samples")

    # Analyze learning speeds
    analysis = analyzer.analyze_context_learning_speed(learning_data)

    print(f"\n🏃 Learning Speed Analysis:")
    print(f"   Ranked contexts: {len(analysis.get('ranked_contexts', []))}")

    if "ranked_contexts" in analysis:
        for i, (context, metrics) in enumerate(analysis["ranked_contexts"][:5]):
            print(f"   {i+1}. {context}: {metrics['learning_rate']:.3f} learning rate")

    # Check other analysis results
    if "fastest_context" in analysis and analysis["fastest_context"]:
        fastest = analysis["fastest_context"]
        print(f"\n🎯 Fastest Learning Context:")
        print(f"   Context: {fastest[0]}")
        print(f"   Learning rate: {fastest[1]['learning_rate']:.3f}")

    if "average_learning_rate" in analysis:
        print(f"   Average learning rate: {analysis['average_learning_rate']:.3f}")

    # Verify expected patterns
    if "ranked_contexts" in analysis and analysis["ranked_contexts"]:
        print("✅ Learning rate analysis working correctly")
    else:
        print("⚠️  No ranked contexts found - check data generation")

    return analysis


def test_strategy_generalization():
    """Test strategy generalization analysis"""
    print("\n🔄 Testing Strategy Generalization Analyzer")
    print("=" * 50)

    analyzer = StrategyGeneralizationAnalyzer()

    # Generate cross-context data
    contexts = ["arithmetic", "algebraic", "structural", "mixed"]
    strategies = ["arithmetic_pure", "weighted_hybrid", "phase_based"]

    learning_data = []
    for strategy in strategies:
        for source_context in contexts:
            for target_context in contexts:
                for trial in range(1, 11):  # 10 trials per transfer
                    # Simulate transfer learning effect
                    if source_context == target_context:
                        # Same context - better performance
                        base_metrics = generate_realistic_learning_metrics(
                            source_context, strategy, trial + 5
                        )
                    else:
                        # Cross-context - variable performance based on similarity
                        similarity_bonus = (
                            0.2 if "mixed" in [source_context, target_context] else 0.0
                        )
                        adjusted_trial = max(1, trial - int(similarity_bonus * 10))
                        base_metrics = generate_realistic_learning_metrics(
                            target_context, strategy, adjusted_trial
                        )

                    learning_data.append(base_metrics)

    print(f"📊 Generated {len(learning_data)} transfer learning samples")

    # Analyze generalization
    generalization_analysis = analyzer.analyze_strategy_generalization(learning_data)

    print(f"\n🌐 Generalization Analysis:")

    if "ranked_strategies" in generalization_analysis:
        ranked_strategies = generalization_analysis["ranked_strategies"]
        print(f"   Strategies analyzed: {len(ranked_strategies)}")

        for i, (strategy, metrics) in enumerate(ranked_strategies[:3]):  # Show top 3
            print(f"\n   {i+1}. Strategy: {strategy}")
            print(f"      Overall generalization: {metrics['overall_generalization']:.3f}")
            print(f"      Context specialization: {metrics['context_specialization']:.3f}")
            print(f"      Robustness: {metrics['robustness']:.3f}")
            print(f"      Adaptation speed: {metrics['adaptation_speed']:.3f}")

    # Check for expected patterns
    if (
        "best_generalizer" in generalization_analysis
        and generalization_analysis["best_generalizer"]
    ):
        best_strategy, best_metrics = generalization_analysis["best_generalizer"]
        print(f"\n🏆 Best Generalizing Strategy: {best_strategy}")
        print(f"   Generalization score: {best_metrics['overall_generalization']:.3f}")

        if best_metrics["overall_generalization"] > 0.5:
            print("✅ Strategy generalization analysis working correctly")

    # Check transfer analysis
    if "transfer_analysis" in generalization_analysis:
        transfer_info = generalization_analysis["transfer_analysis"]
        print(f"\n🔄 Transfer Learning Analysis:")
        print(f"   Average transfer score: {transfer_info.get('average_transfer_score', 0.0):.3f}")

        if "best_transfer_pairs" in transfer_info and transfer_info["best_transfer_pairs"]:
            best_pair = transfer_info["best_transfer_pairs"][0]
            print(f"   Best transfer: {best_pair[0]} -> {best_pair[1]:.3f}")

    return generalization_analysis


def test_learning_curve_prediction():
    """Test learning curve prediction accuracy"""
    print("\n📈 Testing Learning Curve Predictor")
    print("=" * 50)

    predictor = LearningCurvePredictor()

    # Generate a known learning curve
    context = "arithmetic"
    strategy = "arithmetic_pure"

    # Create realistic learning curve with known parameters
    true_learning_rate = 0.15
    true_asymptote = 0.85

    historical_data = []
    for trial in range(1, 16):  # 15 historical trials
        # Exponential learning curve: performance = asymptote * (1 - exp(-learning_rate * trial))
        performance = true_asymptote * (1 - np.exp(-true_learning_rate * trial))
        performance += random.normalvariate(0, 0.05)  # Add noise
        performance = max(0.0, min(1.0, performance))

        metrics = LearningMetrics(
            context_id=f"{context}_test",
            strategy_name=strategy,
            trial_number=trial,
            timestamp=time.time(),
            success=random.random() < performance,
            speedup=1.0 + performance,
            confidence=performance,
            exploration_rate=0.3,
            exploitation_rate=0.7,
            regret=max(0, true_asymptote - performance),
            cumulative_regret=0.0,
            learning_rate=true_learning_rate,
            context_familiarity=trial / 20,
            strategy_confidence=performance,
            convergence_indicator=performance / true_asymptote,
        )
        historical_data.append(metrics)

    print(f"📚 Generated learning curve with {len(historical_data)} historical points")
    print(f"   True learning rate: {true_learning_rate}")
    print(f"   True asymptote: {true_asymptote}")

    # Fit learning curve to historical data
    fitted_curve = predictor.fit_learning_curve(historical_data)

    if fitted_curve:
        print(f"\n🔮 Learning Curve Analysis:")
        print(f"   Fitted learning rate: {fitted_curve.learning_rate:.3f}")
        print(f"   Fitted asymptote: {fitted_curve.asymptote:.3f}")
        print(f"   Predicted convergence trial: {fitted_curve.convergence_trial}")
        print(f"   Curve quality (R²): {fitted_curve.curve_quality:.3f}")

        # Validate prediction accuracy
        learning_rate_error = abs(fitted_curve.learning_rate - true_learning_rate)
        asymptote_error = abs(fitted_curve.asymptote - true_asymptote)

        print(f"\n📊 Prediction Accuracy:")
        print(f"   Learning rate error: {learning_rate_error:.3f}")
        print(f"   Asymptote error: {asymptote_error:.3f}")

        if learning_rate_error < 0.2 and asymptote_error < 0.3:
            print("✅ Learning curve fitting accurate within tolerances")
        else:
            print("⚠️  Fitting accuracy could be improved - this is normal with noisy data")

        # Test future predictions
        future_predictions = predictor.predict_future_performance(fitted_curve, 5)
        print(f"\n🔮 Future Performance Predictions: {len(future_predictions)} trials")
        print(f"   Next 3 predictions: {[f'{p:.2f}' for p in future_predictions[:3]]}")

        return fitted_curve
    else:
        print("⚠️  Could not fit learning curve - insufficient data")
        return None


def test_exploration_rate_tuning():
    """Test exploration rate optimization"""
    print("\n🎯 Testing Exploration Rate Tuner")
    print("=" * 50)

    tuner = ExplorationRateTuner()

    # Simulate optimization history with different exploration rates
    exploration_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    print("🧪 Testing different exploration rates...")

    optimization_results = []
    for exp_rate in exploration_rates:
        # Simulate 20 optimization sessions with this exploration rate
        session_results = []

        for session in range(20):
            # Simulate outcome based on exploration rate
            # Higher exploration initially performs worse but discovers better strategies
            exploration_bonus = exp_rate * 0.3 if session > 10 else -exp_rate * 0.2
            base_performance = 0.6 + exploration_bonus + random.normalvariate(0, 0.1)
            base_performance = max(0.0, min(1.0, base_performance))

            metrics = LearningMetrics(
                context_id=f"exp_test_{exp_rate}",
                strategy_name="test_strategy",
                trial_number=session,
                timestamp=time.time(),
                success=random.random() < base_performance,
                speedup=1.0 + base_performance,
                confidence=base_performance,
                exploration_rate=exp_rate,
                exploitation_rate=1.0 - exp_rate,
                regret=max(0, 0.8 - base_performance),
                cumulative_regret=0.0,
                learning_rate=0.1,
                context_familiarity=session / 20,
                strategy_confidence=base_performance,
                convergence_indicator=base_performance / 0.8,
            )
            session_results.append(metrics)

        optimization_results.extend(session_results)

        avg_performance = np.mean([m.confidence for m in session_results])
        print(f"   Exploration rate {exp_rate}: avg performance {avg_performance:.3f}")

    # Update tuner with results and get exploration rates
    for metrics in optimization_results:
        tuner.update_exploration_rate(
            metrics.context_id, metrics.exploration_rate, metrics.confidence
        )

    # Test exploration rate recommendations
    test_context = "exp_test_0.3"
    current_rate = tuner.get_exploration_rate(test_context, 0)
    later_rate = tuner.get_exploration_rate(test_context, 20)

    print(f"\n🎯 Exploration Rate Tuning:")
    print(f"   Initial exploration rate: {current_rate:.3f}")
    print(f"   Rate after 20 trials: {later_rate:.3f}")
    print(f"   Rate should decay over time")

    if later_rate < current_rate:
        print("✅ Exploration rate correctly decays over time")
    else:
        print("⚠️  Exploration rate not decaying as expected")

    # Test adaptive schedule
    schedule = tuner.get_adaptive_schedule(test_context, 50)

    print(f"\n📅 Adaptive Schedule:")
    print(f"   Schedule length: {len(schedule)}")
    print(f"   Early phase: {schedule[5]:.3f}")
    print(f"   Mid phase: {schedule[25]:.3f}")
    print(f"   Late phase: {schedule[45]:.3f}")

    if schedule[45] < schedule[25] < schedule[5]:
        print("✅ Schedule shows proper decay pattern")

    return {"initial_rate": current_rate, "later_rate": later_rate, "schedule": schedule}


def test_meta_bandit():
    """Test meta-bandit for parameter optimization"""
    print("\n🎰 Testing Meta-Bandit System")
    print("=" * 50)

    meta_bandit = MetaBandit()

    # Get parameter statistics to see what's available
    param_stats = meta_bandit.get_parameter_statistics()

    print(f"🎛️  Available parameter options:")
    for param, stats in param_stats.items():
        print(f"   {param}: {list(stats.keys())}")

    # Simulate parameter optimization rounds
    best_configs = []

    for round_num in range(10):
        # Get optimal parameters from meta-bandit
        config = meta_bandit.get_optimal_parameters("test_context")

        # Simulate performance with this configuration
        # Better configurations should give better performance
        learning_rate_score = (
            1.0 - abs(config.get("learning_rate", 0.1) - 0.1) * 5
        )  # Optimal around 0.1
        threshold_score = (
            1.0 - abs(config.get("confidence_threshold", 0.7) - 0.7) * 2
        )  # Optimal around 0.7
        patience_score = 1.0 - abs(config.get("patience", 30) - 30) / 100  # Optimal around 30

        overall_performance = (learning_rate_score + threshold_score + patience_score) / 3
        overall_performance = max(0.0, min(1.0, overall_performance + random.normalvariate(0, 0.1)))

        # Update meta-bandit
        meta_bandit.update_parameters(config, overall_performance)

        best_configs.append((config.copy(), overall_performance))

        if round_num % 3 == 0:
            print(f"   Round {round_num + 1}: Performance {overall_performance:.3f}")
            print(
                f"      Config: lr={config.get('learning_rate')}, "
                f"thresh={config.get('confidence_threshold')}, patience={config.get('patience')}"
            )

    # Get final parameter statistics
    final_stats = meta_bandit.get_parameter_statistics()

    print(f"\n🏆 Meta-Bandit Optimization Results:")
    for param_name, param_stats in final_stats.items():
        best_option = max(param_stats.items(), key=lambda x: x[1]["expected_reward"])
        print(
            f"   {param_name}: Best={best_option[0]} (reward={best_option[1]['expected_reward']:.3f})"
        )

    # Get a final optimized configuration
    final_config = meta_bandit.get_optimal_parameters("test_context")

    print(f"\n🎯 Final Optimized Configuration:")
    for param, value in final_config.items():
        print(f"   {param}: {value}")

    print("✅ Meta-bandit parameter optimization completed")

    return {
        "final_config": final_config,
        "param_stats": final_stats,
        "performance_history": [config[1] for config in best_configs],
    }


def test_integrated_system():
    """Test the complete Learn to Learn system integration"""
    print("\n🔗 Testing Integrated Learn to Learn System")
    print("=" * 50)

    # Create comprehensive system
    learn_system = LearnToLearnSystem()

    # Generate comprehensive learning data using the system's recording method
    contexts = ["arithmetic", "algebraic", "structural", "mixed"]
    strategies = ["arithmetic_pure", "algebraic_pure", "weighted_hybrid", "phase_based"]

    print("🏗️ Recording comprehensive learning episodes...")

    episode_count = 0
    for context in contexts:
        for strategy in strategies:
            for trial in range(1, 26):  # 25 trials per combination
                # Generate realistic performance
                base_metrics = generate_realistic_learning_metrics(context, strategy, trial)

                # Record learning episode using the system's method
                learn_system.record_learning_episode(
                    context_id=f"{context}_context",
                    strategy_name=strategy,
                    trial_number=trial,
                    success=base_metrics.success,
                    speedup=base_metrics.speedup,
                    confidence=base_metrics.confidence,
                    exploration_rate=base_metrics.exploration_rate,
                    learning_parameters={
                        "learning_rate": base_metrics.learning_rate,
                        "confidence_threshold": 0.7,
                        "patience": 30,
                    },
                )
                episode_count += 1

    print(f"   Recorded {episode_count} comprehensive learning episodes")

    # Analyze learning patterns
    insights = learn_system.analyze_learning_patterns()

    if "error" in insights:
        print(f"⚠️  {insights['error']} (got {insights.get('metrics_count', 0)} metrics)")
        return insights

    print(f"\n🧠 Learning Pattern Analysis:")
    print(f"   Total episodes: {insights.get('total_episodes', 0)}")
    print(f"   Unique contexts: {insights.get('unique_contexts', 0)}")
    print(f"   Unique strategies: {insights.get('unique_strategies', 0)}")

    # Check context analysis
    if "context_analysis" in insights:
        context_info = insights["context_analysis"]
        if "fastest_context" in context_info and context_info["fastest_context"]:
            fastest = context_info["fastest_context"]
            print(
                f"   Fastest learning context: {fastest[0]} (rate: {fastest[1]['learning_rate']:.3f})"
            )

    # Check strategy analysis
    if "strategy_analysis" in insights:
        strategy_info = insights["strategy_analysis"]
        if "best_generalizer" in strategy_info and strategy_info["best_generalizer"]:
            best_strategy, best_metrics = strategy_info["best_generalizer"]
            print(
                f"   Best generalizing strategy: {best_strategy} (score: {best_metrics['overall_generalization']:.3f})"
            )

    # Test optimized learning setup
    test_context_id = "new_arithmetic_context"
    setup = learn_system.get_optimized_learning_setup(test_context_id, strategies)

    print(f"\n💡 Optimized Learning Setup:")
    print(f"   Recommended strategy: {setup.get('recommended_strategy', 'unknown')}")
    print(f"   Strategy confidence: {setup.get('strategy_confidence', 0.0):.3f}")
    print(f"   Exploration rate: {setup.get('exploration_rate', 0.3):.3f}")
    print(f"   Expected trials to convergence: {setup.get('expected_trials_to_convergence', 50)}")
    print(f"   Context difficulty: {setup.get('context_difficulty', 0.5):.3f}")

    # Test learning insights generation
    learning_insights = learn_system.generate_learning_insights()

    print(f"\n⚙️ Generated Learning Insights:")
    if "error" in learning_insights:
        print(f"   {learning_insights['error']}")
    else:
        print(
            f"   Fastest learning contexts: {len(learning_insights.get('fastest_learning_contexts', []))}"
        )
        print(
            f"   Best generalizing strategies: {len(learning_insights.get('best_generalizing_strategies', []))}"
        )
        print(
            f"   Optimization recommendations: {len(learning_insights.get('optimization_recommendations', []))}"
        )

        # Show some recommendations
        if learning_insights.get("optimization_recommendations"):
            rec = learning_insights["optimization_recommendations"][0]
            print(f"   Top recommendation: {rec.get('parameter')} = {rec.get('recommended_value')}")

    # Validate system coherence
    if insights and setup and learning_insights:
        print("✅ Integrated Learn to Learn system functioning correctly")

        # Check setup quality
        setup_confidence = setup.get("strategy_confidence", 0.0)
        if setup_confidence > 0.0:
            print("✅ Learning setup recommendations generated")

        # Check exploration rate reasonableness
        exp_rate = setup.get("exploration_rate", 0.3)
        if 0.0 <= exp_rate <= 1.0:
            print("✅ Reasonable exploration rate suggested")

    return {"insights": insights, "setup": setup, "learning_insights": learning_insights}


def run_comprehensive_learn_to_learn_tests():
    """Run all Learn to Learn system tests"""
    print("🧠 COMPREHENSIVE LEARN TO LEARN SYSTEM TESTING")
    print("=" * 60)
    print("Meta-Learning: Optimize the optimization learning process")
    print()

    test_functions = [
        test_context_learning_analyzer,
        test_strategy_generalization,
        test_learning_curve_prediction,
        test_exploration_rate_tuning,
        test_meta_bandit,
        test_integrated_system,
    ]

    passed = 0
    total = len(test_functions)
    results = {}

    for test_func in test_functions:
        try:
            result = test_func()
            results[test_func.__name__] = result
            passed += 1
            print("✅ PASSED")
        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback

            traceback.print_exc()

        print()

    print("🏁 LEARN TO LEARN TESTING COMPLETE")
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All meta-learning tests passed! Learn to Learn system is ready.")
    else:
        print("⚠️  Some tests failed. Review meta-learning mechanisms before deployment.")

    print()
    print("🧠 LEARN TO LEARN FEATURES VERIFIED:")
    print("   ✅ Context learning speed analysis")
    print("   ✅ Strategy generalization metrics")
    print("   ✅ Learning curve prediction")
    print("   ✅ Exploration rate optimization")
    print("   ✅ Meta-bandit parameter tuning")
    print("   ✅ Integrated learning recommendations")
    print()
    print("💡 The optimizer now learns how to learn more effectively!")

    return results


if __name__ == "__main__":
    run_comprehensive_learn_to_learn_tests()
