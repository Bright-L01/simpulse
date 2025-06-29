"""
Advanced optimization strategies for domain-aware and adaptive optimization.

This module implements sophisticated optimization strategies that learn from
successful patterns and adapt to different mathematical domains.
"""

import asyncio
import json
import logging
import pickle
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import statistics
import math

from ..evolution.models import SimpRule, MutationSuggestion, OptimizationResult, AppliedMutation
from ..evolution.models import MutationType, SimpDirection, SimpPriority
from ..claude.claude_code_client import ClaudeCodeClient

logger = logging.getLogger(__name__)


class MathematicalDomain(Enum):
    """Mathematical domains with different optimization characteristics."""
    ALGEBRA = "algebra"
    TOPOLOGY = "topology"
    ANALYSIS = "analysis"
    CATEGORY_THEORY = "category_theory"
    LOGIC = "logic"
    GEOMETRY = "geometry"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    UNKNOWN = "unknown"


@dataclass
class DomainProfile:
    """Profile of a mathematical domain for optimization."""
    domain: MathematicalDomain
    confidence: float
    characteristics: Dict[str, Any] = field(default_factory=dict)
    rule_patterns: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    dependency_depth: int = 0
    typical_proof_length: float = 0.0
    
    def __post_init__(self):
        # Domain-specific characteristics
        domain_configs = {
            MathematicalDomain.ALGEBRA: {
                'priority_range': (900, 1100),
                'prefer_direction': SimpDirection.POST,
                'complexity_tolerance': 0.8,
                'mutation_aggressiveness': 0.6
            },
            MathematicalDomain.TOPOLOGY: {
                'priority_range': (800, 1200),
                'prefer_direction': SimpDirection.PRE,
                'complexity_tolerance': 0.9,
                'mutation_aggressiveness': 0.4
            },
            MathematicalDomain.ANALYSIS: {
                'priority_range': (850, 1150),
                'prefer_direction': SimpDirection.BOTH,
                'complexity_tolerance': 0.7,
                'mutation_aggressiveness': 0.7
            },
            MathematicalDomain.CATEGORY_THEORY: {
                'priority_range': (700, 1300),
                'prefer_direction': SimpDirection.BOTH,
                'complexity_tolerance': 0.95,
                'mutation_aggressiveness': 0.3
            }
        }
        
        if self.domain in domain_configs:
            self.characteristics.update(domain_configs[self.domain])


@dataclass
class PatternLearning:
    """Learned patterns from successful optimizations."""
    pattern_id: str
    success_rate: float
    contexts: List[str]
    mutations: List[Dict[str, Any]]
    confidence: float
    last_updated: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    
    def update_success(self, successful: bool):
        """Update success rate with new observation."""
        total = self.usage_count + 1
        if successful:
            self.success_rate = (self.success_rate * self.usage_count + 1.0) / total
        else:
            self.success_rate = (self.success_rate * self.usage_count) / total
        
        self.usage_count = total
        self.last_updated = datetime.now()
        
        # Update confidence based on usage
        self.confidence = min(0.95, 0.5 + (self.usage_count * 0.05))


class DomainAwareStrategy:
    """Optimization strategy that adapts to mathematical domains."""
    
    def __init__(self, claude_client: Optional[ClaudeCodeClient] = None):
        """Initialize domain-aware strategy.
        
        Args:
            claude_client: Claude client for intelligent analysis
        """
        self.claude_client = claude_client
        self.domain_patterns = self._initialize_domain_patterns()
        self.keyword_mappings = self._build_keyword_mappings()
        
    def _initialize_domain_patterns(self) -> Dict[MathematicalDomain, Dict[str, Any]]:
        """Initialize domain-specific optimization patterns."""
        return {
            MathematicalDomain.ALGEBRA: {
                'keywords': ['ring', 'group', 'field', 'module', 'algebra', 'hom', 'iso', 'equiv'],
                'rule_patterns': [
                    r'.*_mul_.*', r'.*_add_.*', r'.*_zero.*', r'.*_one.*', 
                    r'.*_inv.*', r'.*_neg.*', r'.*_assoc.*', r'.*_comm.*'
                ],
                'priority_boost': 100,
                'prefer_post_simp': True,
                'chain_length_preference': 'medium'
            },
            MathematicalDomain.TOPOLOGY: {
                'keywords': ['continuous', 'open', 'closed', 'compact', 'metric', 'uniform', 'filter'],
                'rule_patterns': [
                    r'.*continuous.*', r'.*open.*', r'.*closed.*', r'.*compact.*',
                    r'.*_nhds.*', r'.*_closure.*', r'.*_interior.*'
                ],
                'priority_boost': 50,
                'prefer_post_simp': False,
                'chain_length_preference': 'long'
            },
            MathematicalDomain.ANALYSIS: {
                'keywords': ['differentiable', 'integrable', 'measurable', 'limit', 'derivative'],
                'rule_patterns': [
                    r'.*_deriv.*', r'.*_integral.*', r'.*_measure.*', r'.*_limit.*',
                    r'.*_tendsto.*', r'.*_continuous_at.*'
                ],
                'priority_boost': 75,
                'prefer_post_simp': True,
                'chain_length_preference': 'long'
            },
            MathematicalDomain.CATEGORY_THEORY: {
                'keywords': ['category', 'functor', 'natural', 'adjoint', 'monad', 'topos'],
                'rule_patterns': [
                    r'.*_comp.*', r'.*_id.*', r'.*_functor.*', r'.*_natural.*',
                    r'.*_adjoint.*', r'.*_unit.*', r'.*_counit.*'
                ],
                'priority_boost': 25,
                'prefer_post_simp': True,
                'chain_length_preference': 'short'
            }
        }
    
    def _build_keyword_mappings(self) -> Dict[str, MathematicalDomain]:
        """Build keyword to domain mappings."""
        mappings = {}
        for domain, config in self.domain_patterns.items():
            for keyword in config['keywords']:
                mappings[keyword.lower()] = domain
        return mappings
    
    async def analyze_domain(self, module: str, rules: List[SimpRule]) -> DomainProfile:
        """Detect mathematical domain and characteristics of a module.
        
        Args:
            module: Module name
            rules: List of simp rules in the module
            
        Returns:
            Domain profile with detected characteristics
        """
        logger.info(f"Analyzing domain for module: {module}")
        
        # Keyword-based domain detection
        domain_scores = defaultdict(float)
        total_keywords = 0
        
        # Analyze module name
        module_lower = module.lower()
        for keyword, domain in self.keyword_mappings.items():
            if keyword in module_lower:
                domain_scores[domain] += 2.0
                total_keywords += 2.0
        
        # Analyze rule names and patterns
        rule_patterns = []
        for rule in rules:
            rule_name_lower = rule.rule_name.lower()
            rule_patterns.append(rule_name_lower)
            
            # Check keywords in rule names
            for keyword, domain in self.keyword_mappings.items():
                if keyword in rule_name_lower:
                    domain_scores[domain] += 1.0
                    total_keywords += 1.0
        
        # Pattern matching
        for domain, config in self.domain_patterns.items():
            pattern_matches = 0
            for pattern in config['rule_patterns']:
                for rule_name in rule_patterns:
                    if re.match(pattern, rule_name):
                        pattern_matches += 1
            
            if pattern_matches > 0:
                domain_scores[domain] += pattern_matches * 0.5
                total_keywords += pattern_matches * 0.5
        
        # Determine primary domain
        if total_keywords == 0:
            primary_domain = MathematicalDomain.UNKNOWN
            confidence = 0.0
        else:
            primary_domain = max(domain_scores, key=domain_scores.get)
            confidence = min(0.95, domain_scores[primary_domain] / total_keywords)
        
        # Calculate complexity metrics
        complexity_score = self._calculate_complexity_score(rules)
        dependency_depth = len(module.split('.'))
        
        # Estimate typical proof length
        proof_length = self._estimate_proof_length(rules, primary_domain)
        
        profile = DomainProfile(
            domain=primary_domain,
            confidence=confidence,
            rule_patterns=rule_patterns,
            complexity_score=complexity_score,
            dependency_depth=dependency_depth,
            typical_proof_length=proof_length
        )
        
        logger.info(f"Domain analysis complete: {primary_domain.value} (confidence: {confidence:.2f})")
        return profile
    
    def _calculate_complexity_score(self, rules: List[SimpRule]) -> float:
        """Calculate complexity score based on rule characteristics."""
        if not rules:
            return 0.0
        
        complexity_factors = []
        
        for rule in rules:
            # Rule name length as complexity indicator
            name_complexity = len(rule.rule_name) / 50.0
            
            # Priority as complexity indicator (higher priority = more complex)
            priority_complexity = 0.5
            if rule.simp_priority:
                if rule.simp_priority == SimpPriority.HIGH:
                    priority_complexity = 0.8
                elif rule.simp_priority == SimpPriority.LOW:
                    priority_complexity = 0.3
            
            # Direction as complexity indicator
            direction_complexity = 0.5
            if rule.simp_direction == SimpDirection.BOTH:
                direction_complexity = 0.7
            
            rule_complexity = (name_complexity + priority_complexity + direction_complexity) / 3.0
            complexity_factors.append(min(1.0, rule_complexity))
        
        return statistics.mean(complexity_factors)
    
    def _estimate_proof_length(self, rules: List[SimpRule], domain: MathematicalDomain) -> float:
        """Estimate typical proof length for the domain."""
        base_length = len(rules) * 2.0  # Base estimate
        
        # Domain-specific adjustments
        domain_multipliers = {
            MathematicalDomain.CATEGORY_THEORY: 1.5,
            MathematicalDomain.TOPOLOGY: 1.3,
            MathematicalDomain.ANALYSIS: 1.4,
            MathematicalDomain.ALGEBRA: 1.0,
            MathematicalDomain.LOGIC: 0.8,
            MathematicalDomain.UNKNOWN: 1.0
        }
        
        multiplier = domain_multipliers.get(domain, 1.0)
        return base_length * multiplier
    
    def suggest_domain_mutations(self, profile: DomainProfile, rules: List[SimpRule]) -> List[MutationSuggestion]:
        """Generate domain-specific optimization strategies.
        
        Args:
            profile: Domain profile
            rules: List of simp rules
            
        Returns:
            List of domain-specific mutation suggestions
        """
        logger.info(f"Generating domain-specific mutations for {profile.domain.value}")
        
        mutations = []
        domain_config = self.domain_patterns.get(profile.domain, {})
        
        if not domain_config:
            logger.warning(f"No configuration for domain {profile.domain.value}")
            return mutations
        
        # Priority adjustments based on domain
        priority_boost = domain_config.get('priority_boost', 0)
        prefer_post = domain_config.get('prefer_post_simp', True)
        
        for rule in rules:
            # Priority mutations
            if priority_boost > 0:
                mutations.append(MutationSuggestion(
                    rule_name=rule.rule_name,
                    mutation_type=MutationType.PRIORITY_ADJUSTMENT,
                    old_attribute=rule.full_attribute,
                    new_attribute=self._adjust_priority_for_domain(rule.full_attribute, priority_boost),
                    rationale=f"Domain-specific priority boost for {profile.domain.value}",
                    confidence=profile.confidence * 0.8
                ))
            
            # Direction mutations
            if rule.simp_direction != SimpDirection.BOTH:
                new_direction = SimpDirection.POST if prefer_post else SimpDirection.PRE
                if rule.simp_direction != new_direction:
                    mutations.append(MutationSuggestion(
                        rule_name=rule.rule_name,
                        mutation_type=MutationType.DIRECTION_CHANGE,
                        old_attribute=rule.full_attribute,
                        new_attribute=self._change_direction(rule.full_attribute, new_direction),
                        rationale=f"Domain-preferred direction for {profile.domain.value}",
                        confidence=profile.confidence * 0.6
                    ))
        
        # Pattern-specific mutations
        pattern_mutations = self._generate_pattern_mutations(profile, rules, domain_config)
        mutations.extend(pattern_mutations)
        
        logger.info(f"Generated {len(mutations)} domain-specific mutations")
        return mutations
    
    def _adjust_priority_for_domain(self, attribute: str, boost: int) -> str:
        """Adjust priority attribute with domain-specific boost."""
        if not attribute:
            return f"@[simp {boost}]"
        
        # Extract current priority
        priority_match = re.search(r'@\[simp\s+(\d+)\]', attribute)
        if priority_match:
            current_priority = int(priority_match.group(1))
            new_priority = current_priority + boost
            return re.sub(r'@\[simp\s+\d+\]', f'@[simp {new_priority}]', attribute)
        else:
            # Add priority to existing attribute
            if '@[simp]' in attribute:
                return attribute.replace('@[simp]', f'@[simp {1000 + boost}]')
            else:
                return f"@[simp {1000 + boost}]"
    
    def _change_direction(self, attribute: str, new_direction: SimpDirection) -> str:
        """Change simp direction in attribute."""
        if new_direction == SimpDirection.PRE:
            direction_str = "←"
        elif new_direction == SimpDirection.POST:
            direction_str = "→"
        else:
            direction_str = ""
        
        # Remove existing direction
        attribute = re.sub(r'[←→]', '', attribute)
        
        # Add new direction
        if direction_str and '@[simp' in attribute:
            return attribute.replace('@[simp', f'@[simp {direction_str}')
        
        return attribute
    
    def _generate_pattern_mutations(self, profile: DomainProfile, rules: List[SimpRule], 
                                  domain_config: Dict[str, Any]) -> List[MutationSuggestion]:
        """Generate mutations based on domain patterns."""
        mutations = []
        
        patterns = domain_config.get('rule_patterns', [])
        
        for rule in rules:
            for pattern in patterns:
                if re.match(pattern, rule.rule_name):
                    # Create pattern-specific mutation
                    mutations.append(MutationSuggestion(
                        rule_name=rule.rule_name,
                        mutation_type=MutationType.PRIORITY_ADJUSTMENT,
                        old_attribute=rule.full_attribute,
                        new_attribute=self._adjust_priority_for_domain(rule.full_attribute, 50),
                        rationale=f"Pattern-based optimization for {pattern}",
                        confidence=profile.confidence * 0.7
                    ))
                    break
        
        return mutations


class AdaptiveStrategy:
    """Strategy that learns from successful mutations across runs."""
    
    def __init__(self, history_db: Path):
        """Initialize adaptive strategy with learning database.
        
        Args:
            history_db: Path to store learning patterns
        """
        self.history_db = history_db
        self.history_db.parent.mkdir(parents=True, exist_ok=True)
        
        self.success_patterns: Dict[str, PatternLearning] = {}
        self.mutation_success_history = defaultdict(list)
        self.context_patterns = defaultdict(list)
        
        self._load_patterns()
    
    def _load_patterns(self):
        """Load learned patterns from database."""
        try:
            if self.history_db.exists():
                with open(self.history_db, 'rb') as f:
                    data = pickle.load(f)
                    self.success_patterns = data.get('patterns', {})
                    self.mutation_success_history = data.get('history', defaultdict(list))
                    self.context_patterns = data.get('contexts', defaultdict(list))
                logger.info(f"Loaded {len(self.success_patterns)} learned patterns")
        except Exception as e:
            logger.warning(f"Failed to load patterns: {e}")
            self.success_patterns = {}
    
    def _save_patterns(self):
        """Save learned patterns to database."""
        try:
            data = {
                'patterns': self.success_patterns,
                'history': dict(self.mutation_success_history),
                'contexts': dict(self.context_patterns),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.history_db, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved {len(self.success_patterns)} patterns to database")
        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")
    
    async def learn_from_run(self, result: OptimizationResult):
        """Extract successful patterns from optimization result.
        
        Args:
            result: Completed optimization result
        """
        logger.info(f"Learning from optimization run: {result.improvement_percent:.1f}% improvement")
        
        if not result.best_candidate or not result.best_candidate.mutations:
            logger.info("No mutations to learn from")
            return
        
        # Extract successful mutation patterns
        for mutation in result.best_candidate.mutations:
            pattern_id = self._generate_pattern_id(mutation)
            
            # Create or update pattern
            if pattern_id in self.success_patterns:
                pattern = self.success_patterns[pattern_id]
                pattern.update_success(result.success and result.improvement_percent > 0)
            else:
                pattern = PatternLearning(
                    pattern_id=pattern_id,
                    success_rate=1.0 if result.success else 0.0,
                    contexts=[result.modules[0] if result.modules else "unknown"],
                    mutations=[{
                        'type': mutation.mutation_type.value,
                        'old_attr': mutation.old_attribute,
                        'new_attr': mutation.new_attribute,
                        'rule_pattern': self._extract_rule_pattern(mutation.rule_name)
                    }],
                    confidence=0.5,
                    usage_count=1
                )
                self.success_patterns[pattern_id] = pattern
            
            # Track mutation success history
            self.mutation_success_history[mutation.mutation_type.value].append({
                'success': result.success,
                'improvement': result.improvement_percent,
                'timestamp': datetime.now(),
                'context': result.modules[0] if result.modules else "unknown"
            })
        
        # Learn contextual patterns
        if result.modules:
            module = result.modules[0]
            for mutation in result.best_candidate.mutations:
                self.context_patterns[module].append({
                    'mutation_type': mutation.mutation_type.value,
                    'success': result.success,
                    'improvement': result.improvement_percent,
                    'rule_pattern': self._extract_rule_pattern(mutation.rule_name)
                })
        
        # Cleanup old patterns
        self._cleanup_old_patterns()
        
        # Save updated patterns
        self._save_patterns()
        
        logger.info(f"Learning complete. Total patterns: {len(self.success_patterns)}")
    
    def _generate_pattern_id(self, mutation: AppliedMutation) -> str:
        """Generate unique pattern ID for mutation."""
        rule_pattern = self._extract_rule_pattern(mutation.rule_name)
        return f"{mutation.mutation_type.value}_{rule_pattern}"
    
    def _extract_rule_pattern(self, rule_name: str) -> str:
        """Extract pattern from rule name."""
        # Remove module prefixes
        name_parts = rule_name.split('.')
        base_name = name_parts[-1]
        
        # Extract pattern (preserve structure but generalize specifics)
        pattern = re.sub(r'\d+', 'N', base_name)  # Replace numbers
        pattern = re.sub(r'_[a-z](?=_|$)', '_X', pattern)  # Replace single letters
        
        return pattern
    
    def _cleanup_old_patterns(self):
        """Remove patterns that are too old or have low confidence."""
        cutoff_date = datetime.now() - timedelta(days=30)
        
        patterns_to_remove = []
        for pattern_id, pattern in self.success_patterns.items():
            # Remove if too old and low usage
            if (pattern.last_updated < cutoff_date and 
                pattern.usage_count < 5 and 
                pattern.confidence < 0.3):
                patterns_to_remove.append(pattern_id)
        
        for pattern_id in patterns_to_remove:
            del self.success_patterns[pattern_id]
        
        if patterns_to_remove:
            logger.info(f"Cleaned up {len(patterns_to_remove)} old patterns")
    
    def predict_mutation_success(self, mutation: MutationSuggestion, context: str = "") -> float:
        """Predict success probability for a mutation.
        
        Args:
            mutation: Proposed mutation
            context: Module or context for the mutation
            
        Returns:
            Predicted success probability (0.0 to 1.0)
        """
        # Generate pattern ID for lookup
        mock_applied = AppliedMutation(
            rule_name=mutation.rule_name,
            mutation_type=mutation.mutation_type,
            old_attribute=mutation.old_attribute,
            new_attribute=mutation.new_attribute
        )
        pattern_id = self._generate_pattern_id(mock_applied)
        
        # Base prediction from exact pattern match
        base_prediction = 0.5
        confidence_weight = 0.1
        
        if pattern_id in self.success_patterns:
            pattern = self.success_patterns[pattern_id]
            base_prediction = pattern.success_rate
            confidence_weight = pattern.confidence
        
        # Contextual adjustment
        context_adjustment = self._calculate_context_adjustment(mutation, context)
        
        # Type-based adjustment
        type_adjustment = self._calculate_type_adjustment(mutation.mutation_type)
        
        # Combine predictions
        prediction = (
            base_prediction * confidence_weight +
            context_adjustment * 0.3 +
            type_adjustment * 0.2 +
            0.5 * (1.0 - confidence_weight - 0.3 - 0.2)  # default
        )
        
        return max(0.0, min(1.0, prediction))
    
    def _calculate_context_adjustment(self, mutation: MutationSuggestion, context: str) -> float:
        """Calculate context-based success adjustment."""
        if not context or context not in self.context_patterns:
            return 0.5
        
        context_data = self.context_patterns[context]
        if not context_data:
            return 0.5
        
        # Find similar mutations in this context
        similar_mutations = [
            data for data in context_data
            if data['mutation_type'] == mutation.mutation_type.value
        ]
        
        if not similar_mutations:
            return 0.5
        
        # Calculate average success rate
        success_count = sum(1 for data in similar_mutations if data['success'])
        return success_count / len(similar_mutations)
    
    def _calculate_type_adjustment(self, mutation_type: MutationType) -> float:
        """Calculate mutation type-based success adjustment."""
        if mutation_type.value not in self.mutation_success_history:
            return 0.5
        
        history = self.mutation_success_history[mutation_type.value]
        if not history:
            return 0.5
        
        # Recent history is more relevant
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_history = [h for h in history if h['timestamp'] > recent_cutoff]
        
        if recent_history:
            success_count = sum(1 for h in recent_history if h['success'])
            return success_count / len(recent_history)
        else:
            # Fall back to all history
            success_count = sum(1 for h in history if h['success'])
            return success_count / len(history)
    
    def get_recommended_mutations(self, rules: List[SimpRule], context: str = "", 
                                top_k: int = 10) -> List[MutationSuggestion]:
        """Get top recommended mutations based on learned patterns.
        
        Args:
            rules: Available simp rules
            context: Module context
            top_k: Number of top recommendations
            
        Returns:
            Top recommended mutations
        """
        logger.info(f"Generating {top_k} recommended mutations for context: {context}")
        
        recommendations = []
        
        for rule in rules:
            # Generate candidate mutations based on successful patterns
            for pattern in self.success_patterns.values():
                if pattern.success_rate > 0.6 and pattern.confidence > 0.4:
                    # Try to apply this pattern to the rule
                    for mutation_data in pattern.mutations:
                        mutation = self._create_mutation_from_pattern(rule, mutation_data, pattern)
                        if mutation:
                            success_prob = self.predict_mutation_success(mutation, context)
                            mutation.confidence = success_prob
                            recommendations.append(mutation)
        
        # Sort by predicted success and return top_k
        recommendations.sort(key=lambda m: m.confidence, reverse=True)
        
        logger.info(f"Generated {len(recommendations)} candidate mutations")
        return recommendations[:top_k]
    
    def _create_mutation_from_pattern(self, rule: SimpRule, mutation_data: Dict[str, Any], 
                                    pattern: PatternLearning) -> Optional[MutationSuggestion]:
        """Create mutation suggestion from learned pattern."""
        try:
            mutation_type = MutationType(mutation_data['type'])
            
            # Check if pattern is applicable to this rule
            rule_pattern = self._extract_rule_pattern(rule.rule_name)
            if rule_pattern not in pattern.pattern_id:
                return None
            
            # Create mutation suggestion
            return MutationSuggestion(
                rule_name=rule.rule_name,
                mutation_type=mutation_type,
                old_attribute=rule.full_attribute,
                new_attribute=self._adapt_new_attribute(rule, mutation_data),
                rationale=f"Learned pattern (success rate: {pattern.success_rate:.1%})",
                confidence=pattern.confidence
            )
        except Exception as e:
            logger.warning(f"Failed to create mutation from pattern: {e}")
            return None
    
    def _adapt_new_attribute(self, rule: SimpRule, mutation_data: Dict[str, Any]) -> str:
        """Adapt new attribute from pattern to current rule."""
        # This is a simplified adaptation - in practice, this would be more sophisticated
        old_attr = mutation_data.get('old_attr', '')
        new_attr = mutation_data.get('new_attr', '')
        
        if not old_attr or not new_attr:
            return rule.full_attribute
        
        # Try to apply similar transformation
        if '@[simp' in old_attr and '@[simp' in new_attr:
            # Priority change
            old_priority = re.search(r'@\[simp\s+(\d+)\]', old_attr)
            new_priority = re.search(r'@\[simp\s+(\d+)\]', new_attr)
            
            if old_priority and new_priority:
                delta = int(new_priority.group(1)) - int(old_priority.group(1))
                
                # Apply similar delta to current rule
                current_priority = re.search(r'@\[simp\s+(\d+)\]', rule.full_attribute)
                if current_priority:
                    current_val = int(current_priority.group(1))
                    new_val = current_val + delta
                    return re.sub(r'@\[simp\s+\d+\]', f'@[simp {new_val}]', rule.full_attribute)
        
        return rule.full_attribute
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learned patterns."""
        if not self.success_patterns:
            return {"patterns": 0, "avg_success_rate": 0.0, "total_observations": 0}
        
        total_observations = sum(p.usage_count for p in self.success_patterns.values())
        avg_success_rate = statistics.mean(p.success_rate for p in self.success_patterns.values())
        
        return {
            "patterns": len(self.success_patterns),
            "avg_success_rate": avg_success_rate,
            "total_observations": total_observations,
            "high_confidence_patterns": len([p for p in self.success_patterns.values() if p.confidence > 0.7]),
            "recent_patterns": len([p for p in self.success_patterns.values() 
                                  if p.last_updated > datetime.now() - timedelta(days=7)])
        }