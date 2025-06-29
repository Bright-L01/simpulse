"""Parser for Lean 4 profiler trace output.

This module provides functionality to parse and analyze trace output
from Lean's built-in profiler.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime


logger = logging.getLogger(__name__)


@dataclass
class ProfileEntry:
    """Single profiler entry from Lean trace output."""
    name: str
    elapsed_ms: float
    count: int = 1
    children: List['ProfileEntry'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_time_ms(self) -> float:
        """Total time including children."""
        return self.elapsed_ms + sum(child.total_time_ms for child in self.children)
    
    @property
    def self_time_ms(self) -> float:
        """Time spent in this entry excluding children."""
        children_time = sum(child.elapsed_ms for child in self.children)
        return max(0, self.elapsed_ms - children_time)


@dataclass
class SimpRewriteInfo:
    """Information about a simp rewrite application."""
    theorem: str
    target_before: str
    target_after: str
    success: bool
    time_ms: Optional[float] = None
    location: Optional[str] = None


@dataclass
class ProfileReport:
    """Complete profiling report."""
    entries: List[ProfileEntry]
    total_time_ms: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    simp_rewrites: List[SimpRewriteInfo] = field(default_factory=list)
    
    def get_top_entries(self, n: int = 10, by: str = "elapsed") -> List[ProfileEntry]:
        """Get top n entries by specified metric."""
        if by == "elapsed":
            sorted_entries = sorted(self.entries, key=lambda e: e.elapsed_ms, reverse=True)
        elif by == "count":
            sorted_entries = sorted(self.entries, key=lambda e: e.count, reverse=True)
        elif by == "total":
            sorted_entries = sorted(self.entries, key=lambda e: e.total_time_ms, reverse=True)
        else:
            raise ValueError(f"Unknown sort key: {by}")
        return sorted_entries[:n]


class TraceParser:
    """Parser for Lean profiler trace output."""
    
    # Regex patterns for parsing different trace formats
    PROFILE_PATTERN = re.compile(
        r'\[(?P<name>[^\]]+)\]\s+(?P<time>\d+(?:\.\d+)?)\s*ms'
    )
    
    SIMP_REWRITE_PATTERN = re.compile(
        r'\[Meta\.Tactic\.simp\.rewrite\]\s+(?P<theorem>[^\s]+):\s*'
        r'(?P<before>.+?)\s*==>\s*(?P<after>.+?)(?:\s+\((?P<status>\w+)\))?'
    )
    
    JSON_PROFILE_PATTERN = re.compile(
        r'^\s*\{.*"profiler".*\}\s*$',
        re.MULTILINE | re.DOTALL
    )
    
    def __init__(self):
        """Initialize trace parser."""
        self._entry_stack: List[ProfileEntry] = []
        self._current_entries: List[ProfileEntry] = []
        
    def parse_file(self, file_path: Path) -> ProfileReport:
        """Parse a trace file and return profile report.
        
        Args:
            file_path: Path to trace output file
            
        Returns:
            ProfileReport with parsed data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return self.parse_content(content)
        
    def parse_content(self, content: str) -> ProfileReport:
        """Parse trace content and return profile report.
        
        Args:
            content: Trace output content
            
        Returns:
            ProfileReport with parsed data
        """
        # Check if content is JSON format
        if self._is_json_format(content):
            return self._parse_json_format(content)
        else:
            return self._parse_text_format(content)
            
    def _is_json_format(self, content: str) -> bool:
        """Check if content is in JSON format."""
        return bool(self.JSON_PROFILE_PATTERN.search(content))
        
    def _parse_json_format(self, content: str) -> ProfileReport:
        """Parse JSON-formatted profiler output."""
        try:
            data = json.loads(content)
            entries = self._convert_json_to_entries(data.get('profiler', {}))
            
            return ProfileReport(
                entries=entries,
                total_time_ms=sum(e.total_time_ms for e in entries),
                timestamp=datetime.now(),
                metadata=data.get('metadata', {})
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON profile data: {e}")
            # Fall back to text parsing
            return self._parse_text_format(content)
            
    def _convert_json_to_entries(self, json_data: Dict) -> List[ProfileEntry]:
        """Convert JSON profiler data to ProfileEntry objects."""
        entries = []
        
        for name, data in json_data.items():
            entry = ProfileEntry(
                name=name,
                elapsed_ms=data.get('elapsed_ms', 0.0),
                count=data.get('count', 1),
                metadata=data.get('metadata', {})
            )
            
            # Recursively process children
            if 'children' in data:
                entry.children = self._convert_json_to_entries(data['children'])
                
            entries.append(entry)
            
        return entries
        
    def _parse_text_format(self, content: str) -> ProfileReport:
        """Parse text-formatted trace output."""
        entries = []
        simp_rewrites = []
        total_time = 0.0
        
        # Parse line by line
        for line in content.split('\n'):
            # Try to parse profile entry
            profile_match = self.PROFILE_PATTERN.search(line)
            if profile_match:
                entry = ProfileEntry(
                    name=profile_match.group('name'),
                    elapsed_ms=float(profile_match.group('time'))
                )
                entries.append(entry)
                total_time += entry.elapsed_ms
                continue
                
            # Try to parse simp rewrite
            simp_match = self.SIMP_REWRITE_PATTERN.search(line)
            if simp_match:
                rewrite = SimpRewriteInfo(
                    theorem=simp_match.group('theorem'),
                    target_before=simp_match.group('before').strip(),
                    target_after=simp_match.group('after').strip(),
                    success=simp_match.group('status') != 'failed' if simp_match.group('status') else True
                )
                simp_rewrites.append(rewrite)
                
        # Aggregate entries by name
        aggregated = self._aggregate_entries(entries)
        
        return ProfileReport(
            entries=aggregated,
            total_time_ms=total_time,
            timestamp=datetime.now(),
            simp_rewrites=simp_rewrites
        )
        
    def _aggregate_entries(self, entries: List[ProfileEntry]) -> List[ProfileEntry]:
        """Aggregate profile entries by name."""
        aggregated = defaultdict(lambda: {'time': 0.0, 'count': 0})
        
        for entry in entries:
            key = entry.name
            aggregated[key]['time'] += entry.elapsed_ms
            aggregated[key]['count'] += 1
            
        result = []
        for name, data in aggregated.items():
            result.append(ProfileEntry(
                name=name,
                elapsed_ms=data['time'],
                count=data['count']
            ))
            
        return result
        
    def parse_simp_trace(self, content: str) -> Dict[str, Any]:
        """Parse simp-specific trace output.
        
        Args:
            content: Trace output with simp diagnostics
            
        Returns:
            Dictionary with simp statistics
        """
        stats = {
            'total_rewrites': 0,
            'successful_rewrites': 0,
            'failed_rewrites': 0,
            'unique_theorems': set(),
            'rewrite_details': [],
            'most_used_theorems': defaultdict(int),
            'time_by_theorem': defaultdict(float)
        }
        
        for line in content.split('\n'):
            match = self.SIMP_REWRITE_PATTERN.search(line)
            if match:
                theorem = match.group('theorem')
                success = match.group('status') != 'failed' if match.group('status') else True
                
                stats['total_rewrites'] += 1
                if success:
                    stats['successful_rewrites'] += 1
                else:
                    stats['failed_rewrites'] += 1
                    
                stats['unique_theorems'].add(theorem)
                stats['most_used_theorems'][theorem] += 1
                
                stats['rewrite_details'].append({
                    'theorem': theorem,
                    'before': match.group('before').strip(),
                    'after': match.group('after').strip(),
                    'success': success
                })
                
        # Convert set to list for JSON serialization
        stats['unique_theorems'] = list(stats['unique_theorems'])
        
        # Sort most used theorems
        stats['most_used_theorems'] = dict(
            sorted(stats['most_used_theorems'].items(), 
                   key=lambda x: x[1], reverse=True)
        )
        
        return stats
        
    def generate_summary(self, report: ProfileReport) -> str:
        """Generate a human-readable summary of the profile report.
        
        Args:
            report: ProfileReport to summarize
            
        Returns:
            Summary string
        """
        lines = [
            f"Profile Report - {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Time: {report.total_time_ms:.2f} ms",
            f"Total Entries: {len(report.entries)}",
            ""
        ]
        
        # Top entries by time
        lines.append("Top 10 by elapsed time:")
        for i, entry in enumerate(report.get_top_entries(10, by="elapsed"), 1):
            percentage = (entry.elapsed_ms / report.total_time_ms) * 100 if report.total_time_ms > 0 else 0
            lines.append(f"  {i:2d}. {entry.name}: {entry.elapsed_ms:.2f} ms ({percentage:.1f}%)")
            
        # Simp statistics if available
        if report.simp_rewrites:
            lines.extend([
                "",
                f"Simp Rewrites: {len(report.simp_rewrites)}",
                f"Successful: {sum(1 for r in report.simp_rewrites if r.success)}",
                f"Failed: {sum(1 for r in report.simp_rewrites if not r.success)}"
            ])
            
        return '\n'.join(lines)