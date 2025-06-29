"""Workspace manager for handling parallel evaluation environments.

This module manages isolated workspaces for evaluating mutation candidates
in parallel without interference.
"""

import asyncio
import logging
import shutil
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
import time

from .models_v2 import Workspace

logger = logging.getLogger(__name__)


@dataclass
class WorkspacePool:
    """Pool of available workspaces."""
    workspaces: Dict[str, Workspace]
    available: asyncio.Queue
    max_size: int
    
    def __post_init__(self):
        """Initialize the queue with available workspaces."""
        if not hasattr(self, 'available') or self.available is None:
            self.available = asyncio.Queue(maxsize=self.max_size)
            for workspace in self.workspaces.values():
                self.available.put_nowait(workspace)


class WorkspaceManager:
    """Manages isolated workspaces for parallel mutation evaluation."""
    
    def __init__(self, 
                 base_path: Path,
                 max_workspaces: int = 4,
                 workspace_prefix: str = "simpulse_workspace"):
        """Initialize workspace manager.
        
        Args:
            base_path: Base directory for creating workspaces
            max_workspaces: Maximum number of concurrent workspaces
            workspace_prefix: Prefix for workspace directory names
        """
        self.base_path = Path(base_path)
        self.max_workspaces = max_workspaces
        self.workspace_prefix = workspace_prefix
        self.workspace_pool: Optional[WorkspacePool] = None
        self.source_path: Optional[Path] = None
        self._cleanup_on_exit = True
        
        # Ensure base path exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self, source_path: Path) -> bool:
        """Initialize the workspace pool with copies of the source.
        
        Args:
            source_path: Path to source directory to copy
            
        Returns:
            True if initialization successful
        """
        self.source_path = Path(source_path)
        
        if not self.source_path.exists():
            logger.error(f"Source path does not exist: {self.source_path}")
            return False
            
        logger.info(f"Initializing {self.max_workspaces} workspaces from {self.source_path}")
        
        try:
            # Create workspaces
            workspaces = {}
            available_queue = asyncio.Queue(maxsize=self.max_workspaces)
            
            for i in range(self.max_workspaces):
                workspace_id = f"{self.workspace_prefix}_{i}"
                workspace_path = self.base_path / workspace_id
                
                # Create workspace copy
                if await self._create_workspace_copy(workspace_path):
                    workspace = Workspace(
                        id=workspace_id,
                        path=workspace_path,
                        is_active=False
                    )
                    workspaces[workspace_id] = workspace
                    await available_queue.put(workspace)
                else:
                    logger.error(f"Failed to create workspace {workspace_id}")
                    return False
                    
            self.workspace_pool = WorkspacePool(
                workspaces=workspaces,
                available=available_queue,
                max_size=self.max_workspaces
            )
            
            logger.info(f"Successfully initialized {len(workspaces)} workspaces")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize workspaces: {e}")
            return False
            
    async def acquire_workspace(self, candidate_id: str, timeout: float = 30.0) -> Optional[Workspace]:
        """Acquire an available workspace for evaluation.
        
        Args:
            candidate_id: ID of candidate that will use the workspace
            timeout: Timeout for acquiring workspace
            
        Returns:
            Available workspace or None if timeout
        """
        if not self.workspace_pool:
            logger.error("Workspace pool not initialized")
            return None
            
        try:
            # Wait for available workspace
            workspace = await asyncio.wait_for(
                self.workspace_pool.available.get(),
                timeout=timeout
            )
            
            # Activate workspace
            workspace.activate(candidate_id)
            
            logger.debug(f"Acquired workspace {workspace.id} for candidate {candidate_id}")
            return workspace
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout acquiring workspace for candidate {candidate_id}")
            return None
        except Exception as e:
            logger.error(f"Error acquiring workspace: {e}")
            return None
            
    async def release_workspace(self, workspace: Workspace) -> bool:
        """Release workspace back to the pool.
        
        Args:
            workspace: Workspace to release
            
        Returns:
            True if successfully released
        """
        if not self.workspace_pool:
            logger.error("Workspace pool not initialized")
            return False
            
        try:
            # Clean workspace
            await self._clean_workspace(workspace)
            
            # Deactivate workspace
            workspace.deactivate()
            
            # Return to pool
            await self.workspace_pool.available.put(workspace)
            
            logger.debug(f"Released workspace {workspace.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error releasing workspace {workspace.id}: {e}")
            return False
            
    async def _create_workspace_copy(self, workspace_path: Path) -> bool:
        """Create a workspace copy of the source directory.
        
        Args:
            workspace_path: Path for new workspace
            
        Returns:
            True if copy successful
        """
        try:
            # Remove existing workspace if present
            if workspace_path.exists():
                shutil.rmtree(workspace_path)
                
            # Create workspace using efficient copying
            await self._efficient_copy(self.source_path, workspace_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create workspace copy at {workspace_path}: {e}")
            return False
            
    async def _efficient_copy(self, src: Path, dst: Path) -> None:
        """Efficiently copy directory structure.
        
        Uses hard links where possible for better performance.
        
        Args:
            src: Source directory
            dst: Destination directory
        """
        def copy_with_hardlinks(src_path: Path, dst_path: Path):
            """Copy using hard links where possible."""
            dst_path.mkdir(parents=True, exist_ok=True)
            
            for item in src_path.iterdir():
                src_item = src_path / item.name
                dst_item = dst_path / item.name
                
                if item.is_dir():
                    # Recursively copy directories
                    copy_with_hardlinks(src_item, dst_item)
                else:
                    # Try hard link first, fall back to copy
                    try:
                        os.link(src_item, dst_item)
                    except (OSError, PermissionError):
                        # Hard link failed, use regular copy
                        shutil.copy2(src_item, dst_item)
                        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, copy_with_hardlinks, src, dst)
        
    async def _clean_workspace(self, workspace: Workspace) -> None:
        """Clean workspace by restoring from source.
        
        Args:
            workspace: Workspace to clean
        """
        try:
            # Get list of files that might have been modified
            modified_files = await self._find_modified_files(workspace.path)
            
            if modified_files:
                logger.debug(f"Cleaning {len(modified_files)} modified files in workspace {workspace.id}")
                
                # Restore modified files from source
                for rel_path in modified_files:
                    src_file = self.source_path / rel_path
                    dst_file = workspace.path / rel_path
                    
                    if src_file.exists():
                        shutil.copy2(src_file, dst_file)
                        
        except Exception as e:
            logger.warning(f"Error cleaning workspace {workspace.id}: {e}")
            # If cleaning fails, recreate the entire workspace
            await self._recreate_workspace(workspace)
            
    async def _find_modified_files(self, workspace_path: Path) -> List[Path]:
        """Find files that have been modified in the workspace.
        
        Args:
            workspace_path: Path to workspace
            
        Returns:
            List of relative paths to modified files
        """
        modified = []
        
        def check_modifications(workspace_dir: Path, source_dir: Path, rel_path: Path = Path()):
            """Recursively check for modifications."""
            try:
                for item in workspace_dir.iterdir():
                    if item.name.startswith('.'):
                        continue
                        
                    item_rel_path = rel_path / item.name
                    source_item = source_dir / item.name
                    
                    if item.is_dir() and source_item.is_dir():
                        check_modifications(item, source_item, item_rel_path)
                    elif item.is_file() and source_item.is_file():
                        # Check if file was modified (simple size/mtime check)
                        if (item.stat().st_size != source_item.stat().st_size or
                            abs(item.stat().st_mtime - source_item.stat().st_mtime) > 1):
                            modified.append(item_rel_path)
                            
            except Exception as e:
                logger.debug(f"Error checking modifications in {workspace_dir}: {e}")
                
        # Run in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, check_modifications, workspace_path, self.source_path
        )
        
        return modified
        
    async def _recreate_workspace(self, workspace: Workspace) -> bool:
        """Recreate workspace from scratch.
        
        Args:
            workspace: Workspace to recreate
            
        Returns:
            True if recreation successful
        """
        try:
            logger.info(f"Recreating workspace {workspace.id}")
            
            # Remove existing workspace
            if workspace.path.exists():
                shutil.rmtree(workspace.path)
                
            # Create new copy
            return await self._create_workspace_copy(workspace.path)
            
        except Exception as e:
            logger.error(f"Failed to recreate workspace {workspace.id}: {e}")
            return False
            
    async def cleanup_all(self) -> None:
        """Clean up all workspaces and remove directories."""
        if not self.workspace_pool:
            return
            
        logger.info("Cleaning up all workspaces")
        
        try:
            # Wait for all workspaces to be available (not in use)
            active_workspaces = [w for w in self.workspace_pool.workspaces.values() if w.is_active]
            
            if active_workspaces:
                logger.warning(f"Waiting for {len(active_workspaces)} active workspaces to complete")
                
                # Wait up to 30 seconds for workspaces to become available
                timeout = 30.0
                start_time = time.time()
                
                while active_workspaces and (time.time() - start_time) < timeout:
                    await asyncio.sleep(1.0)
                    active_workspaces = [w for w in self.workspace_pool.workspaces.values() if w.is_active]
                    
            # Remove workspace directories
            for workspace in self.workspace_pool.workspaces.values():
                if workspace.path.exists():
                    try:
                        shutil.rmtree(workspace.path)
                        logger.debug(f"Removed workspace directory {workspace.path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove workspace {workspace.path}: {e}")
                        
            # Clear pool
            self.workspace_pool = None
            
        except Exception as e:
            logger.error(f"Error during workspace cleanup: {e}")
            
    def get_workspace_status(self) -> Dict[str, Any]:
        """Get status information about workspaces.
        
        Returns:
            Dictionary with workspace status
        """
        if not self.workspace_pool:
            return {"status": "not_initialized"}
            
        active_count = sum(1 for w in self.workspace_pool.workspaces.values() if w.is_active)
        available_count = self.workspace_pool.available.qsize()
        
        return {
            "status": "initialized",
            "total_workspaces": len(self.workspace_pool.workspaces),
            "active_workspaces": active_count,
            "available_workspaces": available_count,
            "base_path": str(self.base_path),
            "source_path": str(self.source_path) if self.source_path else None
        }
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        if self._cleanup_on_exit:
            await self.cleanup_all()
            
    def set_cleanup_on_exit(self, cleanup: bool):
        """Set whether to cleanup workspaces on exit.
        
        Args:
            cleanup: Whether to cleanup on exit
        """
        self._cleanup_on_exit = cleanup