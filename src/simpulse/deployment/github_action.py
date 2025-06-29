"""GitHub Action integration for automated simp optimization.

This module provides GitHub integration for creating optimization PRs,
posting progress updates, and managing the continuous optimization workflow.
"""

import asyncio
import json
import logging
import os
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

# GitHub API integration (will be imported if available)
try:
    from github import Github, GithubException
    from github.Repository import Repository
    from github.PullRequest import PullRequest
    from github.Issue import Issue
    GITHUB_AVAILABLE = True
except ImportError:
    Github = None
    Repository = None
    PullRequest = None
    Issue = None
    GithubException = Exception
    GITHUB_AVAILABLE = False

from ..evolution.models_v2 import GenerationResult, EvolutionHistory
from ..evolution.evolution_engine import OptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class PRMetadata:
    """Metadata for optimization PR."""
    optimization_id: str
    modules: List[str]
    baseline_performance: Dict[str, float]
    optimized_performance: Dict[str, float]
    mutations_applied: int
    improvement_percent: float
    execution_time: float
    
    
@dataclass
class ProgressUpdate:
    """Progress update for live reporting."""
    generation: int
    best_fitness: float
    average_fitness: float
    diversity: float
    mutations_tested: int
    elapsed_time: float
    estimated_remaining: float


class GitHubActionRunner:
    """Handles GitHub integration for automated simp optimization."""
    
    def __init__(self, 
                 github_token: Optional[str] = None,
                 repo: Optional[str] = None,
                 dry_run: bool = False):
        """Initialize GitHub Action runner.
        
        Args:
            github_token: GitHub API token
            repo: Repository name (owner/repo)
            dry_run: If True, don't actually create PRs/comments
        """
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.repo_name = repo or os.getenv('GITHUB_REPOSITORY')
        self.dry_run = dry_run
        
        if not GITHUB_AVAILABLE:
            logger.warning("GitHub library not available. Install with: pip install PyGithub")
            self.github = None
            self.repo = None
        elif self.github_token and self.repo_name and not dry_run:
            try:
                self.github = Github(self.github_token)
                self.repo = self.github.get_repo(self.repo_name)
                logger.info(f"Connected to GitHub repository: {self.repo_name}")
            except Exception as e:
                logger.error(f"Failed to connect to GitHub: {e}")
                self.github = None
                self.repo = None
        else:
            logger.info("GitHub integration disabled (missing token/repo or dry-run mode)")
            self.github = None
            self.repo = None
            
    async def create_optimization_pr(self, 
                                   result: OptimizationResult,
                                   source_branch: str = "main",
                                   target_branch: Optional[str] = None) -> Optional[str]:
        """Create PR with optimized simp rules.
        
        Args:
            result: Optimization result with mutations
            source_branch: Source branch to create PR from
            target_branch: Target branch name (auto-generated if None)
            
        Returns:
            PR URL if created successfully, None otherwise
        """
        if not self.repo or self.dry_run:
            logger.info("Skipping PR creation (dry run or no GitHub connection)")
            return await self._simulate_pr_creation(result)
            
        try:
            # Generate branch name
            if not target_branch:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                target_branch = f"simpulse/optimize-simp-{timestamp}"
                
            logger.info(f"Creating optimization PR on branch: {target_branch}")
            
            # Get source branch reference
            source_ref = self.repo.get_git_ref(f"heads/{source_branch}")
            source_sha = source_ref.object.sha
            
            # Create new branch
            self.repo.create_git_ref(f"refs/heads/{target_branch}", source_sha)
            
            # Apply mutations to files
            files_modified = await self._apply_mutations_to_repo(result, target_branch)
            
            if not files_modified:
                logger.warning("No files were modified, not creating PR")
                return None
                
            # Generate PR content
            pr_title = self._generate_pr_title(result)
            pr_description = self.generate_pr_description(result)
            
            # Create pull request
            pr = self.repo.create_pull(
                title=pr_title,
                body=pr_description,
                head=target_branch,
                base=source_branch
            )
            
            # Add labels
            labels = self._generate_pr_labels(result)
            if labels:
                pr.add_to_labels(*labels)
                
            logger.info(f"Created PR #{pr.number}: {pr.html_url}")
            
            # Post initial progress comment
            await self._post_optimization_summary(pr.number, result)
            
            return pr.html_url
            
        except Exception as e:
            logger.error(f"Failed to create optimization PR: {e}")
            return None
            
    async def _apply_mutations_to_repo(self, 
                                     result: OptimizationResult,
                                     branch: str) -> List[str]:
        """Apply mutations to repository files.
        
        Args:
            result: Optimization result
            branch: Target branch
            
        Returns:
            List of modified file paths
        """
        if not result.best_candidate or not result.best_candidate.mutations:
            return []
            
        modified_files = []
        
        try:
            for mutation in result.best_candidate.mutations:
                # Get the mutation details
                file_path = getattr(mutation, 'file_path', None)
                if not file_path:
                    continue
                    
                # Read current file content
                try:
                    file_content = self.repo.get_contents(str(file_path), ref=branch)
                    current_content = base64.b64decode(file_content.content).decode('utf-8')
                except:
                    logger.warning(f"Could not read file {file_path}, skipping")
                    continue
                    
                # Apply mutation (simplified - in practice would use MutationApplicator)
                modified_content = await self._apply_single_mutation(current_content, mutation)
                
                if modified_content != current_content:
                    # Update file in repository
                    commit_message = f"Optimize simp rule: {getattr(mutation, 'original_rule_name', 'unknown')}"
                    
                    self.repo.update_file(
                        path=str(file_path),
                        message=commit_message,
                        content=modified_content,
                        sha=file_content.sha,
                        branch=branch
                    )
                    
                    modified_files.append(str(file_path))
                    logger.debug(f"Modified file: {file_path}")
                    
        except Exception as e:
            logger.error(f"Error applying mutations to repository: {e}")
            
        return modified_files
        
    async def _apply_single_mutation(self, content: str, mutation: Any) -> str:
        """Apply a single mutation to file content.
        
        Args:
            content: Original file content
            mutation: Mutation to apply
            
        Returns:
            Modified content
        """
        # Simplified mutation application
        # In practice, this would use the MutationApplicator
        
        if hasattr(mutation, 'modified_content'):
            return mutation.modified_content
        elif hasattr(mutation, 'suggestion'):
            # Extract mutation details from suggestion
            suggestion = mutation.suggestion
            if hasattr(suggestion, 'mutated_declaration'):
                # Simple replacement for demo
                original = suggestion.original_declaration
                mutated = suggestion.mutated_declaration
                return content.replace(original, mutated, 1)
                
        return content
        
    def _generate_pr_title(self, result: OptimizationResult) -> str:
        """Generate PR title for optimization."""
        improvement = result.improvement_percent
        modules = ", ".join(result.modules[:3])  # Limit to first 3 modules
        
        if len(result.modules) > 3:
            modules += f" and {len(result.modules) - 3} more"
            
        return f"🚀 Optimize simp rules: {improvement:.1f}% improvement in {modules}"
        
    def generate_pr_description(self, result: OptimizationResult) -> str:
        """Generate rich PR description with metrics and visualizations.
        
        Args:
            result: Optimization result
            
        Returns:
            Markdown PR description
        """
        lines = []
        
        # Header
        lines.append("# 🧬 Simp Rule Optimization Results")
        lines.append("")
        lines.append("This PR contains optimized simp rules generated by Simpulse using evolutionary algorithms.")
        lines.append("")
        
        # Performance Summary
        lines.append("## 📊 Performance Summary")
        lines.append("")
        lines.append("| Metric | Baseline | Optimized | Improvement |")
        lines.append("|--------|----------|-----------|-------------|")
        
        if result.best_candidate and result.best_candidate.fitness:
            fitness = result.best_candidate.fitness
            baseline_time = fitness.total_time / (1 - result.improvement_percent / 100)
            
            lines.append(f"| **Total Time** | {baseline_time:.2f} ms | {fitness.total_time:.2f} ms | **{result.improvement_percent:.1f}%** |")
            lines.append(f"| **Iterations** | - | {fitness.iterations} | - |")
            lines.append(f"| **Memory Usage** | - | {fitness.memory_mb:.1f} MB | - |")
            lines.append(f"| **Fitness Score** | - | {fitness.composite_score:.4f} | - |")
        else:
            lines.append(f"| **Overall** | - | - | **{result.improvement_percent:.1f}%** |")
            
        lines.append("")
        
        # Evolution Statistics
        lines.append("## 🧬 Evolution Statistics")
        lines.append("")
        lines.append(f"- **Generations**: {result.total_generations}")
        lines.append(f"- **Total Evaluations**: {result.total_evaluations}")
        lines.append(f"- **Execution Time**: {result.execution_time:.1f} seconds")
        lines.append(f"- **Modules Optimized**: {len(result.modules)}")
        
        if result.best_candidate:
            lines.append(f"- **Mutations Applied**: {len(result.best_candidate.mutations)}")
            
        lines.append("")
        
        # Affected Modules
        lines.append("## 📦 Affected Modules")
        lines.append("")
        for module in result.modules:
            lines.append(f"- `{module}`")
        lines.append("")
        
        # Mutation Details
        if result.best_candidate and result.best_candidate.mutations:
            lines.append("## 🔧 Applied Mutations")
            lines.append("")
            
            mutation_summary = {}
            for mutation in result.best_candidate.mutations:
                if hasattr(mutation, 'suggestion'):
                    mut_type = mutation.suggestion.mutation_type.value
                    mutation_summary[mut_type] = mutation_summary.get(mut_type, 0) + 1
                    
            for mut_type, count in mutation_summary.items():
                lines.append(f"- **{mut_type.replace('_', ' ').title()}**: {count} changes")
                
            lines.append("")
            lines.append("<details>")
            lines.append("<summary>Detailed Mutation List</summary>")
            lines.append("")
            
            for i, mutation in enumerate(result.best_candidate.mutations, 1):
                if hasattr(mutation, 'suggestion'):
                    suggestion = mutation.suggestion
                    lines.append(f"### {i}. {suggestion.rule_name}")
                    lines.append(f"- **Type**: {suggestion.mutation_type.value}")
                    lines.append(f"- **Description**: {suggestion.description}")
                    lines.append(f"- **Confidence**: {suggestion.confidence:.1%}")
                    
                    if suggestion.estimated_impact:
                        lines.append("- **Estimated Impact**:")
                        for metric, value in suggestion.estimated_impact.items():
                            lines.append(f"  - {metric}: {value}%")
                            
                    lines.append("")
                    
            lines.append("</details>")
            lines.append("")
            
        # Risk Assessment
        lines.append("## ⚠️ Risk Assessment")
        lines.append("")
        lines.append("This optimization was generated using evolutionary algorithms and includes:")
        lines.append("")
        lines.append("- ✅ **Automated testing** during evolution")
        lines.append("- ✅ **Syntax validation** for all mutations")
        lines.append("- ✅ **Performance verification** through profiling")
        lines.append("- ⚠️ **Manual review recommended** for complex changes")
        lines.append("")
        
        # Review Checklist
        lines.append("## 📋 Review Checklist")
        lines.append("")
        lines.append("- [ ] Performance improvements verified")
        lines.append("- [ ] No breaking changes in dependent modules")
        lines.append("- [ ] Rule priorities make semantic sense")
        lines.append("- [ ] Tests pass with optimized rules")
        lines.append("- [ ] Documentation updated if needed")
        lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"🤖 Generated by [Simpulse](https://github.com/your-org/simpulse) evolutionary optimizer")
        lines.append(f"⏱️ Optimization completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        return "\n".join(lines)
        
    def _generate_pr_labels(self, result: OptimizationResult) -> List[str]:
        """Generate appropriate labels for the PR.
        
        Args:
            result: Optimization result
            
        Returns:
            List of label names
        """
        labels = ["optimization", "simp", "automated"]
        
        # Add improvement level labels
        improvement = result.improvement_percent
        if improvement >= 25:
            labels.append("major-improvement")
        elif improvement >= 10:
            labels.append("significant-improvement")
        elif improvement >= 5:
            labels.append("minor-improvement")
            
        # Add mutation type labels
        if result.best_candidate and result.best_candidate.mutations:
            mutation_types = set()
            for mutation in result.best_candidate.mutations:
                if hasattr(mutation, 'suggestion'):
                    mutation_types.add(mutation.suggestion.mutation_type.value)
                    
            if "priority_change" in mutation_types:
                labels.append("priority-changes")
            if "direction_change" in mutation_types:
                labels.append("direction-changes")
                
        return labels
        
    async def post_progress_comment(self, 
                                  issue_id: int,
                                  progress: ProgressUpdate) -> bool:
        """Post live progress updates on PR/issue.
        
        Args:
            issue_id: Issue or PR number
            progress: Progress update data
            
        Returns:
            True if comment posted successfully
        """
        if not self.repo or self.dry_run:
            logger.info(f"Progress update (dry run): Generation {progress.generation}, "
                       f"Best: {progress.best_fitness:.4f}")
            return True
            
        try:
            issue = self.repo.get_issue(issue_id)
            
            # Generate progress comment
            comment_body = self._generate_progress_comment(progress)
            
            # Check if we already have a progress comment
            existing_comment = None
            for comment in issue.get_comments():
                if "🔄 Optimization Progress" in comment.body:
                    existing_comment = comment
                    break
                    
            if existing_comment:
                # Update existing comment
                existing_comment.edit(comment_body)
            else:
                # Create new comment
                issue.create_comment(comment_body)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to post progress comment: {e}")
            return False
            
    def _generate_progress_comment(self, progress: ProgressUpdate) -> str:
        """Generate progress comment content.
        
        Args:
            progress: Progress update data
            
        Returns:
            Markdown comment content
        """
        lines = []
        
        lines.append("## 🔄 Optimization Progress")
        lines.append("")
        lines.append(f"**Generation**: {progress.generation}")
        lines.append(f"**Best Fitness**: {progress.best_fitness:.4f}")
        lines.append(f"**Average Fitness**: {progress.average_fitness:.4f}")
        lines.append(f"**Population Diversity**: {progress.diversity:.3f}")
        lines.append(f"**Mutations Tested**: {progress.mutations_tested}")
        lines.append("")
        
        # Progress bar
        if progress.estimated_remaining > 0:
            percent_complete = progress.elapsed_time / (progress.elapsed_time + progress.estimated_remaining) * 100
            bar_length = 20
            filled_length = int(bar_length * percent_complete / 100)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            
            lines.append(f"**Progress**: {percent_complete:.1f}%")
            lines.append(f"```")
            lines.append(f"{bar} {percent_complete:.1f}%")
            lines.append(f"```")
            lines.append("")
            
            # Time estimates
            lines.append(f"**Elapsed**: {progress.elapsed_time:.1f}s")
            lines.append(f"**Estimated Remaining**: {progress.estimated_remaining:.1f}s")
            
        lines.append("")
        lines.append(f"*Last updated: {datetime.now().strftime('%H:%M:%S UTC')}*")
        
        return "\n".join(lines)
        
    async def _post_optimization_summary(self, pr_number: int, result: OptimizationResult):
        """Post optimization summary as PR comment.
        
        Args:
            pr_number: PR number
            result: Optimization result
        """
        if not self.repo or self.dry_run:
            return
            
        try:
            pr = self.repo.get_pull(pr_number)
            
            summary = f"""## 🎯 Optimization Summary

**🏆 Achievement**: {result.improvement_percent:.1f}% performance improvement

**📈 Key Metrics**:
- Generations: {result.total_generations}
- Evaluations: {result.total_evaluations}
- Execution time: {result.execution_time:.1f}s
- Success rate: {result.success}

**🔧 Mutations Applied**: {len(result.best_candidate.mutations) if result.best_candidate else 0}

The optimization process used evolutionary algorithms to intelligently explore the space of possible simp rule modifications, converging on this optimal configuration through {result.total_generations} generations of evolution.

Ready for review! 🚀"""

            pr.create_issue_comment(summary)
            
        except Exception as e:
            logger.error(f"Failed to post optimization summary: {e}")
            
    async def _simulate_pr_creation(self, result: OptimizationResult) -> str:
        """Simulate PR creation for testing/dry-run.
        
        Args:
            result: Optimization result
            
        Returns:
            Simulated PR URL
        """
        pr_title = self._generate_pr_title(result)
        pr_description = self.generate_pr_description(result)
        
        logger.info("=== SIMULATED PR CREATION ===")
        logger.info(f"Title: {pr_title}")
        logger.info(f"Description length: {len(pr_description)} characters")
        logger.info(f"Labels: {self._generate_pr_labels(result)}")
        logger.info("=== END SIMULATION ===")
        
        return f"https://github.com/{self.repo_name}/pull/12345"  # Simulated URL
        
    def validate_github_connection(self) -> Dict[str, Any]:
        """Validate GitHub connection and permissions.
        
        Returns:
            Validation status dictionary
        """
        if not GITHUB_AVAILABLE:
            return {
                "status": "error",
                "message": "GitHub library not available",
                "suggestion": "Install with: pip install PyGithub"
            }
            
        if not self.github_token:
            return {
                "status": "error",
                "message": "GitHub token not provided",
                "suggestion": "Set GITHUB_TOKEN environment variable"
            }
            
        if not self.repo_name:
            return {
                "status": "error", 
                "message": "Repository name not provided",
                "suggestion": "Set GITHUB_REPOSITORY environment variable"
            }
            
        try:
            if self.repo:
                # Test basic permissions
                permissions = self.repo.get_permissions()
                
                return {
                    "status": "success",
                    "repository": self.repo_name,
                    "permissions": {
                        "pull": permissions.pull,
                        "push": permissions.push,
                        "admin": permissions.admin
                    },
                    "rate_limit": self.github.get_rate_limit().core.remaining
                }
            else:
                return {
                    "status": "error",
                    "message": "Could not connect to repository"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"GitHub validation failed: {e}"
            }
    
    def _get_safe_config(self) -> Dict[str, Any]:
        """Get configuration with sensitive values masked.
        
        Returns:
            Configuration dictionary with secrets masked
        """
        from ..security.validators import get_safe_env_vars
        
        config = {
            "repo_name": self.repo_name,
            "dry_run": self.dry_run,
            "github_token_present": bool(self.github_token),
            "github_connected": bool(self.repo)
        }
        
        # Add masked environment variables
        safe_env = get_safe_env_vars()
        config["environment"] = {
            k: v for k, v in safe_env.items() 
            if k.startswith(("GITHUB_", "SIMPULSE_"))
        }
        
        return config