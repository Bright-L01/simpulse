# AlphaEvolve-Style Simp Rule Optimizer for Lean 4
## Complete Design & Implementation Guide

### Executive Summary

An evolutionary optimization system that automatically tunes `simp` tactic performance in Lean 4, initially targeting mathlib4. Using AlphaEvolve-inspired techniques, it discovers optimal rule priorities and configurations to achieve 20-50% compilation speedup while maintaining proof correctness.

---

## 1. System Architecture

### 1.1 Core Components

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Lean Profiler     │────▶│ Evolution Engine │────▶│ Patch Generator │
│ - trace.profiler    │     │ - Population mgmt│     │ - Git patches   │
│ - diagnostics       │     │ - Claude mutations│    │ - PR creation   │
│ - timing extraction │     │ - Fitness eval   │     │ - Reports       │
└─────────────────────┘     └──────────────────┘     └─────────────────┘
           ▲                          │                         │
           │                          ▼                         ▼
    ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
    │ Rule Extractor│         │   Evaluator    │         │GitHub Action │
    │ - AST parsing │         │ - Lake builds  │         │ - Scheduling │
    │ - Simp rules  │         │ - Validation   │         │ - Artifacts  │
    │ - Dependencies│         │ - Metrics      │         │ - Reporting  │
    └──────────────┘         └──────────────┘         └──────────────┘
```

### 1.2 Data Flow

1. **Baseline Profiling**: Extract current simp performance metrics
2. **Rule Analysis**: Parse all `@[simp]` declarations and their contexts  
3. **Evolution Loop**: Generate, evaluate, and select rule configurations
4. **Validation**: Ensure all proofs still pass with modified rules
5. **Deployment**: Create PR with optimized configuration

---

## 2. Technical Specifications

### 2.1 Profiling Infrastructure

```python
# Profiling command structure
lake env lean -Dtrace.profiler.output=prof.json \
             -Dtrace.profiler=true \
             -Ddiagnostics=true \
             -Ddiagnostics.threshold=10 \
             Module.lean
```

**Key Metrics Extraction**:
- Total simp time per goal
- Number of simp iterations
- Rule application frequency
- Memory allocation patterns
- Proof search depth

### 2.2 Search Space Definition

```python
class SimpRuleConfig:
    rule_name: str
    priority: int  # 100-10000, default 1000
    phase: Literal["pre", "post", "both"]  # When to apply
    enabled: bool  # Whether to include in simp set
    module: str    # Original module location
    
class ModuleConfig:
    module_name: str
    rules: List[SimpRuleConfig]
    custom_simp_sets: Dict[str, List[str]]  # Named simp sets
```

### 2.3 Evolution Strategy

```python
class EvolutionParameters:
    population_size: int = 30
    elite_size: int = 3  # Top 10% preserved
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    
    # Adaptive parameters
    stagnation_threshold: int = 10  # Generations without improvement
    mutation_amplification: float = 1.5  # Increase on stagnation
    
    # Multi-objective weights
    time_weight: float = 0.6
    iteration_weight: float = 0.2
    depth_weight: float = 0.15
    memory_weight: float = 0.05
```

### 2.4 Mutation Operations

1. **Priority Mutations**:
   - Small: ±100 (fine-tuning)
   - Medium: ±500 (reordering within tier)
   - Large: ±1000 (tier changes)

2. **Structural Mutations**:
   - Phase switch (pre ↔ post)
   - Rule enable/disable
   - Custom set membership

3. **Intelligent Mutations** (Claude-guided):
   - Context-aware priority suggestions
   - Mathematical domain clustering
   - Dependency-based reordering

---

## 3. Implementation Phases

### Phase 0: Infrastructure (Days 0-3)
**Deliverables**:
- Profiler integration with JSON parsing
- Baseline timing database schema
- Rule extraction from Lean AST
- Initial metric collection

**Key Files**:
```
src/lean_simp_opt/
├── profiler.py       # Trace data extraction
├── rule_extractor.py # AST parsing for @[simp]
├── database.py       # SQLite for metrics
└── metrics.py        # Statistical analysis
```

### Phase 1: Claude Integration (Days 4-8)
**Deliverables**:
- Prompt templates for rule analysis
- Mutation generation system
- Syntax validation
- Priority suggestion engine

**Prompt Template Example**:
```
CONTEXT: Analyzing simp performance for Mathlib.Algebra.Group.Basic
SLOWEST GOALS:
1. `mul_inv_cancel`: 347ms, 89 iterations
2. `inv_mul_cancel`: 312ms, 76 iterations

CURRENT RULES (top 10 by usage):
- mul_one [priority: 1000]: used 2341 times
- one_mul [priority: 1000]: used 2298 times
- mul_inv_cancel [priority: 1000]: used 187 times

Suggest 5 priority adjustments to reduce iterations.
Format: rule_name, new_priority, reasoning
```

### Phase 2: Evolution Engine (Days 9-15)
**Deliverables**:
- Population management with diversity
- Parallel fitness evaluation
- Tournament selection
- Adaptive mutation rates

**Evaluation Pipeline**:
```python
async def evaluate_candidate(config: ModuleConfig) -> Fitness:
    # 1. Write modified @[simp] attributes
    patch = generate_patch(config)
    
    # 2. Run parallel lake build
    result = await lake_build_async(
        modules=[config.module_name],
        jobs=cpu_count(),
        timeout=300
    )
    
    # 3. Extract metrics if successful
    if result.success:
        return extract_fitness_metrics(result.profile_data)
    else:
        return Fitness(valid=False)
```

### Phase 3: CI Integration (Days 16-21)
**Deliverables**:
- GitHub Action workflow
- Docker container setup
- Progress visualization
- Automated PR generation

**Workflow Configuration**:
```yaml
name: Optimize Simp Rules
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily
  workflow_dispatch:
    inputs:
      modules:
        description: 'Modules to optimize (comma-separated)'
        default: 'Mathlib.Algebra,Mathlib.Topology'
      time_budget:
        description: 'Time budget in seconds'
        default: '7200'  # 2 hours
      target_improvement:
        description: 'Target improvement percentage'
        default: '20'

jobs:
  optimize:
    runs-on: [self-hosted, lean-optimizer]  # Requires 16GB RAM
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/lean-simp-optimizer
        with:
          claude_api_key: ${{ secrets.CLAUDE_API_KEY }}
          modules: ${{ inputs.modules }}
          time_budget: ${{ inputs.time_budget }}
```

### Phase 4: Release & Monitoring (Days 22-28)
**Deliverables**:
- Performance dashboard
- Documentation & tutorials
- Community feedback system
- Continuous improvement pipeline

---

## 4. Safety & Validation

### 4.1 Correctness Guarantees
- **Invariant**: All proofs must still succeed
- **Validation**: Full `lake build` on modified modules
- **Rollback**: Automatic reversion on failure

### 4.2 Performance Safeguards
- **Regression Detection**: Alert if >5% slowdown
- **Stability Metrics**: Track proof fragility
- **Incremental Rollout**: Test on small modules first

### 4.3 Review Process
1. Automated PR with detailed metrics
2. Diff visualization for rule changes
3. Performance comparison graphs
4. Community review period (72 hours)

---

## 5. Optimization Strategy

### 5.1 Module Prioritization
1. **High-Impact First**: Start with most-used modules
   - `Mathlib.Algebra.Group`
   - `Mathlib.Data.Set.Basic`
   - `Mathlib.Topology.Basic`

2. **Domain Clustering**: Optimize related modules together
   - Algebraic structures
   - Topological spaces
   - Category theory

### 5.2 Search Heuristics
- **Frequency-weighted mutations**: Prioritize frequently-used rules
- **Dependency-aware ordering**: Respect proof dependencies
- **Domain-specific tuning**: Different strategies per mathematical area

### 5.3 Multi-Objective Optimization
```python
def fitness_function(metrics: Metrics) -> float:
    return (
        0.60 * (baseline.total_time / metrics.total_time) +
        0.20 * (baseline.iterations / metrics.iterations) +
        0.15 * (baseline.max_depth / metrics.max_depth) +
        0.05 * (baseline.memory / metrics.memory)
    )
```

---

## 6. Expected Outcomes

### 6.1 Performance Targets
- **Primary**: 20-50% reduction in simp time
- **Secondary**: 15-30% fewer simp iterations
- **Stretch**: 10% overall compilation speedup

### 6.2 Success Metrics
- Adoption by 5+ mathlib4 contributors
- 100+ GitHub stars within 3 months
- Integration into mathlib4 CI pipeline
- Measurable impact on development velocity

### 6.3 Future Extensions
- Cross-repository optimization
- Real-time optimization during development
- Integration with other tactics (ring, field_simp)
- Machine learning for pattern recognition

---

## 7. Resource Requirements

### 7.1 Development Environment
- 16GB+ RAM for mathlib4 compilation
- 8+ CPU cores for parallel evaluation
- 100GB storage for caching
- Claude API access (Opus preferred)

### 7.2 CI Infrastructure
- Self-hosted runners recommended
- Nightly optimization runs (2-4 hours)
- Artifact storage for historical data
- Monitoring and alerting system

---

## 8. Risk Mitigation

### 8.1 Technical Risks
- **Risk**: Non-deterministic simp behavior
- **Mitigation**: Multiple evaluation runs, statistical validation

- **Risk**: Breaking downstream projects
- **Mitigation**: Extensive testing, gradual rollout

### 8.2 Resource Risks
- **Risk**: Excessive compute costs
- **Mitigation**: Budget limits, incremental optimization

- **Risk**: API rate limiting
- **Mitigation**: Request queuing, fallback strategies

---

## 9. Implementation Timeline

**Week 1**: Infrastructure & Profiling
**Week 2**: Evolution Engine & Claude Integration  
**Week 3**: CI/CD & Validation Systems
**Week 4**: Polish, Documentation & Launch

---

## 10. Code Structure

```
lean-simp-optimizer/
├── src/
│   ├── profiling/
│   │   ├── trace_parser.py
│   │   ├── metrics_extractor.py
│   │   └── baseline_generator.py
│   ├── evolution/
│   │   ├── population.py
│   │   ├── mutations.py
│   │   ├── selection.py
│   │   └── fitness.py
│   ├── claude/
│   │   ├── prompts.py
│   │   ├── api_client.py
│   │   └── suggestion_parser.py
│   ├── evaluation/
│   │   ├── lake_runner.py
│   │   ├── validation.py
│   │   └── metrics_collector.py
│   └── deployment/
│       ├── patch_generator.py
│       ├── pr_creator.py
│       └── report_builder.py
├── tests/
├── examples/
├── .github/
│   ├── workflows/
│   └── actions/
├── Dockerfile
├── requirements.txt
└── README.md
```

This design provides a solid foundation for building an effective simp rule optimizer while maintaining flexibility for future enhancements and community contributions.
