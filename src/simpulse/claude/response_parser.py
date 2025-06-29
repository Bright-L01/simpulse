"""Parser for Claude Code CLI responses.

This module parses Claude's responses and extracts structured
mutation suggestions and analysis results.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..evolution.models import MutationSuggestion, MutationType

logger = logging.getLogger(__name__)


@dataclass
class ParsedSuggestion:
    """Intermediate representation of a parsed suggestion."""
    rule_name: str
    mutation_type: str
    description: str
    original_declaration: str
    mutated_declaration: str
    reasoning: str
    confidence: float
    estimated_impact: Dict[str, float]
    risks: List[str]
    prerequisites: List[str]


class ResponseParser:
    """Parses Claude Code responses for mutation suggestions."""
    
    def __init__(self):
        """Initialize response parser."""
        self.json_pattern = re.compile(
            r'```json\s*(\{.*?\})\s*```', 
            re.DOTALL | re.MULTILINE
        )
        
        # Patterns for extracting structured data from text responses
        self.suggestion_patterns = {
            'rule_name': re.compile(r'(?:Rule|Rule Name):\s*([^\n]+)', re.IGNORECASE),
            'mutation_type': re.compile(r'(?:Mutation Type|Type):\s*([^\n]+)', re.IGNORECASE),
            'description': re.compile(r'(?:Description|Change):\s*([^\n]+)', re.IGNORECASE),
            'reasoning': re.compile(r'(?:Reasoning|Why):\s*([^\n]+(?:\n(?!\w+:).+)*)', re.IGNORECASE | re.MULTILINE),
            'confidence': re.compile(r'(?:Confidence):\s*(\d+)%?', re.IGNORECASE)
        }
        
    def parse_mutations(self, claude_output: str) -> List[MutationSuggestion]:
        """Extract structured mutations from Claude's response.
        
        Args:
            claude_output: Raw output from Claude Code CLI
            
        Returns:
            List of parsed MutationSuggestion objects
        """
        suggestions = []
        
        # Try JSON parsing first
        json_suggestions = self._parse_json_response(claude_output)
        if json_suggestions:
            suggestions.extend(json_suggestions)
        
        # If no JSON found or incomplete, try text parsing
        if not suggestions:
            text_suggestions = self._parse_text_response(claude_output)
            suggestions.extend(text_suggestions)
            
        logger.info(f"Parsed {len(suggestions)} mutation suggestions")
        return suggestions
        
    def _parse_json_response(self, output: str) -> List[MutationSuggestion]:
        """Parse JSON-formatted response.
        
        Args:
            output: Claude's output containing JSON
            
        Returns:
            List of parsed suggestions
        """
        suggestions = []
        
        # Find JSON blocks
        json_matches = self.json_pattern.findall(output)
        
        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                
                if "suggestions" in data:
                    for suggestion_data in data["suggestions"]:
                        suggestion = self._create_mutation_from_dict(suggestion_data)
                        if suggestion:
                            suggestions.append(suggestion)
                            
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                continue
                
        return suggestions
        
    def _parse_text_response(self, output: str) -> List[MutationSuggestion]:
        """Parse text-formatted response.
        
        Args:
            output: Claude's text output
            
        Returns:
            List of parsed suggestions
        """
        suggestions = []
        
        # Split into sections that might contain suggestions
        sections = self._split_into_suggestion_sections(output)
        
        for section in sections:
            suggestion = self._parse_text_suggestion(section)
            if suggestion:
                suggestions.append(suggestion)
                
        return suggestions
        
    def _split_into_suggestion_sections(self, text: str) -> List[str]:
        """Split text into potential suggestion sections.
        
        Args:
            text: Input text
            
        Returns:
            List of text sections
        """
        # Look for numbered suggestions or clear section breaks
        sections = []
        
        # Try splitting by numbered items (1., 2., etc.)
        numbered_pattern = re.compile(r'\n\s*\d+\.\s*', re.MULTILINE)
        numbered_sections = numbered_pattern.split(text)
        
        if len(numbered_sections) > 1:
            # Skip the first section (before first number)
            sections.extend(numbered_sections[1:])
        else:
            # Try other section markers
            markers = ['##', '**Suggestion', '**Rule', 'Mutation:']
            
            for marker in markers:
                if marker in text:
                    parts = text.split(marker)
                    sections.extend(parts[1:])  # Skip first empty part
                    break
            else:
                # Fallback: treat whole text as one section
                sections = [text]
                
        return [s.strip() for s in sections if s.strip()]
        
    def _parse_text_suggestion(self, section_text: str) -> Optional[MutationSuggestion]:
        """Parse a single suggestion from text section.
        
        Args:
            section_text: Text section containing one suggestion
            
        Returns:
            MutationSuggestion object or None
        """
        try:
            # Extract basic fields using patterns
            rule_name = self._extract_field(section_text, 'rule_name', 'Unknown')
            mutation_type_str = self._extract_field(section_text, 'mutation_type', 'unknown')
            description = self._extract_field(section_text, 'description', '')
            reasoning = self._extract_field(section_text, 'reasoning', '')
            confidence_str = self._extract_field(section_text, 'confidence', '50')
            
            # Parse confidence
            try:
                confidence = float(confidence_str.replace('%', '')) / 100.0
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
            except (ValueError, AttributeError):
                confidence = 0.5
                
            # Parse mutation type
            try:
                mutation_type = self._parse_mutation_type(mutation_type_str)
            except ValueError:
                logger.warning(f"Unknown mutation type: {mutation_type_str}")
                mutation_type = MutationType.PRIORITY_CHANGE  # Default
                
            # Extract code blocks for original and mutated declarations
            original_decl, mutated_decl = self._extract_code_blocks(section_text)
            
            # Extract lists (risks, prerequisites)
            risks = self._extract_list_field(section_text, ['risk', 'risks', 'concern'])
            prerequisites = self._extract_list_field(section_text, ['prerequisite', 'requirement'])
            
            # Extract estimated impact
            estimated_impact = self._extract_impact_estimates(section_text)
            
            return MutationSuggestion(
                rule_name=rule_name,
                mutation_type=mutation_type,
                description=description,
                original_declaration=original_decl,
                mutated_declaration=mutated_decl,
                reasoning=reasoning,
                confidence=confidence,
                estimated_impact=estimated_impact,
                risks=risks,
                prerequisites=prerequisites
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse text suggestion: {e}")
            return None
            
    def _extract_field(self, text: str, field_name: str, default: str = '') -> str:
        """Extract a field value from text using patterns.
        
        Args:
            text: Input text
            field_name: Field to extract
            default: Default value if not found
            
        Returns:
            Extracted field value
        """
        if field_name in self.suggestion_patterns:
            match = self.suggestion_patterns[field_name].search(text)
            if match:
                return match.group(1).strip()
                
        return default
        
    def _extract_code_blocks(self, text: str) -> Tuple[str, str]:
        """Extract original and mutated code blocks from text.
        
        Args:
            text: Input text containing code blocks
            
        Returns:
            Tuple of (original_code, mutated_code)
        """
        # Look for code blocks marked with ```lean or ```
        code_block_pattern = re.compile(r'```(?:lean)?\s*(.*?)```', re.DOTALL)
        code_blocks = code_block_pattern.findall(text)
        
        original_decl = ""
        mutated_decl = ""
        
        if len(code_blocks) >= 2:
            original_decl = code_blocks[0].strip()
            mutated_decl = code_blocks[1].strip()
        elif len(code_blocks) == 1:
            # Try to determine if it's original or mutated based on context
            if "original" in text.lower() or "before" in text.lower():
                original_decl = code_blocks[0].strip()
            else:
                mutated_decl = code_blocks[0].strip()
                
        # If no code blocks, try to extract from inline code
        if not original_decl and not mutated_decl:
            inline_code_pattern = re.compile(r'`([^`]+)`')
            inline_codes = inline_code_pattern.findall(text)
            
            if len(inline_codes) >= 2:
                original_decl = inline_codes[0]
                mutated_decl = inline_codes[1]
                
        return original_decl, mutated_decl
        
    def _extract_list_field(self, text: str, keywords: List[str]) -> List[str]:
        """Extract list items for fields like risks or prerequisites.
        
        Args:
            text: Input text
            keywords: Keywords to look for
            
        Returns:
            List of extracted items
        """
        items = []
        
        for keyword in keywords:
            # Look for bullet points after keyword
            pattern = re.compile(
                f'{keyword}s?:([^\\n]*(?:\\n[-*•]\\s*[^\\n]+)*)', 
                re.IGNORECASE | re.MULTILINE
            )
            
            match = pattern.search(text)
            if match:
                content = match.group(1)
                
                # Extract bullet points
                bullet_pattern = re.compile(r'[-*•]\s*([^\n]+)')
                bullet_items = bullet_pattern.findall(content)
                
                if bullet_items:
                    items.extend([item.strip() for item in bullet_items])
                else:
                    # Try comma-separated items
                    comma_items = [item.strip() for item in content.split(',') if item.strip()]
                    items.extend(comma_items)
                    
                break
                
        return items
        
    def _extract_impact_estimates(self, text: str) -> Dict[str, float]:
        """Extract estimated impact percentages from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of impact estimates
        """
        impact = {}
        
        # Look for percentage improvements
        percentage_patterns = {
            'time_improvement_percent': re.compile(r'time.*?(\d+)%', re.IGNORECASE),
            'memory_impact_percent': re.compile(r'memory.*?(\d+)%', re.IGNORECASE),
            'success_rate_change': re.compile(r'success.*?(\d+)%', re.IGNORECASE)
        }
        
        for impact_type, pattern in percentage_patterns.items():
            match = pattern.search(text)
            if match:
                try:
                    value = float(match.group(1))
                    impact[impact_type] = value
                except ValueError:
                    continue
                    
        return impact
        
    def _parse_mutation_type(self, type_str: str) -> MutationType:
        """Parse mutation type from string.
        
        Args:
            type_str: String representation of mutation type
            
        Returns:
            MutationType enum value
            
        Raises:
            ValueError: If type string is not recognized
        """
        type_str = type_str.lower().strip()
        
        # Mapping of common strings to mutation types
        type_mappings = {
            'priority_change': MutationType.PRIORITY_CHANGE,
            'priority': MutationType.PRIORITY_CHANGE,
            'condition_add': MutationType.CONDITION_ADD,
            'add_condition': MutationType.CONDITION_ADD,
            'condition_remove': MutationType.CONDITION_REMOVE,
            'remove_condition': MutationType.CONDITION_REMOVE,
            'condition_modify': MutationType.CONDITION_MODIFY,
            'modify_condition': MutationType.CONDITION_MODIFY,
            'pattern_simplify': MutationType.PATTERN_SIMPLIFY,
            'simplify': MutationType.PATTERN_SIMPLIFY,
            'pattern_generalize': MutationType.PATTERN_GENERALIZE,
            'generalize': MutationType.PATTERN_GENERALIZE,
            'direction_change': MutationType.DIRECTION_CHANGE,
            'direction': MutationType.DIRECTION_CHANGE,
            'rule_split': MutationType.RULE_SPLIT,
            'split': MutationType.RULE_SPLIT,
            'rule_combine': MutationType.RULE_COMBINE,
            'combine': MutationType.RULE_COMBINE,
            'rule_disable': MutationType.RULE_DISABLE,
            'disable': MutationType.RULE_DISABLE
        }
        
        if type_str in type_mappings:
            return type_mappings[type_str]
            
        # Try partial matching
        for key, value in type_mappings.items():
            if type_str in key or key in type_str:
                return value
                
        raise ValueError(f"Unknown mutation type: {type_str}")
        
    def _create_mutation_from_dict(self, data: Dict[str, Any]) -> Optional[MutationSuggestion]:
        """Create MutationSuggestion from dictionary data.
        
        Args:
            data: Dictionary containing suggestion data
            
        Returns:
            MutationSuggestion object or None if creation fails
        """
        try:
            # Parse mutation type
            mutation_type_str = data.get('mutation_type', 'priority_change')
            try:
                mutation_type = self._parse_mutation_type(mutation_type_str)
            except ValueError:
                logger.warning(f"Unknown mutation type in JSON: {mutation_type_str}")
                mutation_type = MutationType.PRIORITY_CHANGE
                
            # Parse confidence
            confidence = data.get('confidence', 50)
            if isinstance(confidence, str):
                confidence = float(confidence.replace('%', ''))
            confidence = float(confidence) / 100.0 if confidence > 1 else float(confidence)
            confidence = max(0.0, min(1.0, confidence))
            
            return MutationSuggestion(
                rule_name=data.get('rule_name', 'Unknown'),
                mutation_type=mutation_type,
                description=data.get('description', ''),
                original_declaration=data.get('original_declaration', ''),
                mutated_declaration=data.get('mutated_declaration', ''),
                reasoning=data.get('reasoning', ''),
                confidence=confidence,
                estimated_impact=data.get('estimated_impact', {}),
                risks=data.get('risks', []),
                prerequisites=data.get('prerequisites', [])
            )
            
        except Exception as e:
            logger.warning(f"Failed to create mutation from dict: {e}")
            return None
            
    def parse_analysis_response(self, claude_output: str) -> Dict[str, Any]:
        """Parse general analysis response from Claude.
        
        Args:
            claude_output: Raw output from Claude Code CLI
            
        Returns:
            Parsed analysis results
        """
        analysis = {
            "bottlenecks": [],
            "opportunities": [],
            "recommendations": [],
            "summary": ""
        }
        
        # Try to extract structured sections
        sections = self._extract_analysis_sections(claude_output)
        
        analysis.update(sections)
        analysis["raw_output"] = claude_output
        
        return analysis
        
    def _extract_analysis_sections(self, text: str) -> Dict[str, Any]:
        """Extract structured sections from analysis response.
        
        Args:
            text: Analysis response text
            
        Returns:
            Dictionary of extracted sections
        """
        sections = {}
        
        # Define section patterns
        section_patterns = {
            'bottlenecks': re.compile(r'(?:performance\s+)?bottlenecks?:?\s*(.*?)(?=\n\s*\w+:|$)', re.IGNORECASE | re.DOTALL),
            'opportunities': re.compile(r'optimization\s+opportunities?:?\s*(.*?)(?=\n\s*\w+:|$)', re.IGNORECASE | re.DOTALL),
            'recommendations': re.compile(r'recommendations?:?\s*(.*?)(?=\n\s*\w+:|$)', re.IGNORECASE | re.DOTALL)
        }
        
        for section_name, pattern in section_patterns.items():
            match = pattern.search(text)
            if match:
                content = match.group(1).strip()
                # Extract bullet points or numbered items
                items = self._extract_items_from_text(content)
                sections[section_name] = items
                
        return sections
        
    def _extract_items_from_text(self, text: str) -> List[str]:
        """Extract list items from text (bullets, numbers, etc.).
        
        Args:
            text: Input text
            
        Returns:
            List of extracted items
        """
        items = []
        
        # Try bullet points first
        bullet_pattern = re.compile(r'[-*•]\s*([^\n]+)')
        bullet_items = bullet_pattern.findall(text)
        
        if bullet_items:
            items.extend([item.strip() for item in bullet_items])
        else:
            # Try numbered items
            numbered_pattern = re.compile(r'\d+\.\s*([^\n]+)')
            numbered_items = numbered_pattern.findall(text)
            
            if numbered_items:
                items.extend([item.strip() for item in numbered_items])
            else:
                # Split by newlines and filter
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                items.extend(lines)
                
        return items