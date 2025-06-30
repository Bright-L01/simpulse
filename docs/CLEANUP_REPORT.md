SIMPULSE CODEBASE CLEANUP REPORT
======================================================================

Total Files: 33
Total Functions: 316

ISSUES FOUND:
----------------------------------------------------------------------

1. UNUSED FUNCTIONS (95)
   - cli_v2.version
   - cli_v2.serve
   - cli_v2.async_wrapper
   - cli_v2.report
   - cli_v2.validate
   - config.get_cache_path
   - config._validate_api_key
   - config.is_claude_available
   - config.save_to_file
   - config.get_config
   ... and 85 more

3. DUPLICATE CONSTANTS (185)
   - '--time-budget' in 2 modules
   - '--population-size' in 2 modules
   - '--cache-dir' in 2 modules
   - 'Time budget in seconds' in 2 modules
   - 'claude_code' in 4 modules

4. LARGE FILES (26)
   - /Users/brightliu/Coding_Projects/simpulse/src/simpulse/cli_v2.py: 534 lines
   - /Users/brightliu/Coding_Projects/simpulse/src/simpulse/config.py: 451 lines
   - /Users/brightliu/Coding_Projects/simpulse/src/simpulse/reporting/report_generator.py: 785 lines
   - /Users/brightliu/Coding_Projects/simpulse/src/simpulse/core/refactor.py: 534 lines
   - /Users/brightliu/Coding_Projects/simpulse/src/simpulse/analysis/impact_analyzer.py: 679 lines

5. COMPLEX FUNCTIONS (5)
   - security.validators.validate_command_args: complexity 12
   - security.validators.validate_json_structure: complexity 12
   - web.dashboard.setup_routes: complexity 16
   - claude.prompt_builder._build_performance_section: complexity 11
   - deployment.github_action.generate_pr_description: complexity 12

RECOMMENDATIONS:
----------------------------------------------------------------------
1. Refactor large files: ['/Users/brightliu/Coding_Projects/simpulse/src/simpulse/cli_v2.py', '/Users/brightliu/Coding_Projects/simpulse/src/simpulse/config.py', '/Users/brightliu/Coding_Projects/simpulse/src/simpulse/reporting/report_generator.py']
2. Simplify complex functions: ['validate_command_args', 'validate_json_structure', 'setup_routes']
3. Focus on core functionality: rule extraction, mutation, and profiling
4. Remove experimental features until core is proven
5. Consolidate similar modules (e.g., models.py and models_v2.py)
6. Create clear separation between CLI, core logic, and utilities

IMMEDIATE ACTIONS:
----------------------------------------------------------------------
1. Remove all unused functions and imports
2. Extract duplicate constants to a shared constants.py
3. Split evolution_engine.py (400+ lines) into smaller modules
4. Merge models.py and models_v2.py
5. Focus on core loop: profile → extract → mutate → measure

MODULES TO CONSIDER REMOVING:
----------------------------------------------------------------------
- web/dashboard.py (not essential for core functionality)
- strategies/advanced_strategies.py (premature optimization)
- deployment/* (until core is proven)
- benchmarks/* (until we have real results)

This cleanup could reduce codebase by ~40% while maintaining core functionality.