"""
Log file collectors for MERIT monitoring.

This module provides collectors that can extract LLM interaction data
from log files in various formats.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Pattern
import re
from pathlib import Path
import gzip
import threading

from .collector import BaseDataCollector, CollectionResult, CollectionStatus
from ..models import LLMRequest, LLMResponse, LLMInteraction, TokenInfo


class LogDataCollector(BaseDataCollector):
    """
    Collector that extracts LLM interaction data from log files.
    
    This collector can monitor log files in various formats and extract
    structured information about LLM interactions. It supports JSON logs,
    as well as custom log formats through regex patterns.
    
    This can work in both:
    - Batch mode: Process existing log files at once
    - Tail mode: Watch log files for new entries (like 'tail -f')
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the log file collector.
        
        Args:
            config: Configuration dictionary with the following options:
                - log_path: Path to log file or directory of log files
                - file_pattern: Glob pattern for log files (default: "*.log")
                - format: Format of log entries ("json" or "regex")
                - regex_pattern: Regex pattern for extracting data (if format="regex")
                - json_path: JSON path to LLM data (if format="json")
                - tail: Whether to watch files for new data (default: False)
                - tail_interval: How often to check for new data in seconds (default: 1.0)
                - batch_size: Number of log entries to process at once (default: 100)
                - max_history: Max number of historical log entries to process (default: 1000)
        """
        super().__init__(config)
        self.log_path = config.get("log_path", "")
        self.file_pattern = config.get("file_pattern", "*.log")
        self.format = config.get("format", "json")
        self.regex_pattern = config.get("regex_pattern", "")
        self.json_path = config.get("json_path", "")  # e.g., "llm.interaction"
        self.tail = config.get("tail", False)
        self.tail_interval = config.get("tail_interval", 1.0)
        self.batch_size = config.get("batch_size", 100)
        self.max_history = config.get("max_history", 1000)
        
        # Compiled regex pattern if needed
        self._compiled_regex: Optional[Pattern] = None
        if self.format == "regex" and self.regex_pattern:
            self._compiled_regex = re.compile(self.regex_pattern)
        
        # State for file monitoring
        self._file_positions: Dict[str, int] = {}  # file path -> last position
        self._processed_files: Set[str] = set()  # files that have been fully processed
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """
        Start collecting data from log files.
        
        If in tail mode, this starts a background thread to monitor log files
        for new data. Otherwise, it just prepares the collector for batch operations.
        """
        super().start()
        
        # Initialize file positions for monitoring
        if self.tail:
            # Start from end of files for tail mode
            for file_path in self._get_log_files():
                if os.path.isfile(file_path):
                    self._file_positions[file_path] = os.path.getsize(file_path)
            
            # Start monitoring thread
            if not self._monitor_thread or not self._monitor_thread.is_alive():
                self._monitor_thread = threading.Thread(
                    target=self._monitor_files, 
                    daemon=True
                )
                self._monitor_thread.start()
    
    def stop(self) -> None:
        """
        Stop collecting data from log files.
        
        This stops any background monitoring threads and cleans up resources.
        """
        super().stop()
        
        # Wait for monitor thread to finish if it's running
        if self._monitor_thread and self._monitor_thread.is_alive():
            # We can't directly stop the thread, but setting is_running to False
            # will cause it to exit on the next iteration
            self._monitor_thread.join(timeout=self.tail_interval * 2)
    
    def collect(self) -> CollectionResult:
        """
        Process log files and extract LLM interaction data.
        
        In batch mode, this processes all matching log files up to max_history.
        In tail mode with no background thread, this processes new entries since
        the last call.
        
        Returns:
            CollectionResult with the extracted data
        """
        start_time = datetime.now()
        result = CollectionResult(
            status=CollectionStatus.SUCCESS,
            start_time=start_time
        )
        
        try:
            # Get list of log files to process
            log_files = self._get_log_files()
            
            for file_path in log_files:
                if not os.path.isfile(file_path):
                    continue
                    
                # Skip already processed files in batch mode
                if not self.tail and file_path in self._processed_files:
                    continue
                
                # Get position to start reading from
                position = self._file_positions.get(file_path, 0)
                
                # Process the file
                new_position, file_result = self._process_file(
                    file_path, position, self.batch_size
                )
                
                # Update state
                self._file_positions[file_path] = new_position
                if new_position >= os.path.getsize(file_path):
                    self._processed_files.add(file_path)
                
                # Update result
                result.items_processed += file_result.items_processed
                result.items_collected += file_result.items_collected
                result.data.extend(file_result.data)
                
                if result.items_processed >= self.max_history:
                    break
            
            # Determine final status
            if result.items_processed == 0:
                result.status = CollectionStatus.SUCCESS  # No data is still success
            elif result.items_collected == 0:
                result.status = CollectionStatus.FAILURE
                result.error = "No valid LLM interactions found in logs"
            elif result.items_collected < result.items_processed:
                result.status = CollectionStatus.PARTIAL
                result.error = f"Only processed {result.items_collected}/{result.items_processed} items"
        
        except Exception as e:
            result.status = CollectionStatus.FAILURE
            result.error = str(e)
        
        finally:
            result.end_time = datetime.now()
            return result
    
    def _get_log_files(self) -> List[str]:
        """
        Get list of log files to process based on configuration.
        
        Returns:
            List of file paths to process
        """
        if not self.log_path:
            return []
            
        log_path = Path(self.log_path)
        
        if log_path.is_file():
            return [str(log_path)]
            
        if log_path.is_dir():
            # Get files matching the pattern
            files = list(log_path.glob(self.file_pattern))
            # Sort by modification time (newest first)
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            return [str(f) for f in files]
            
        return []
    
    def _process_file(self, file_path: str, start_position: int, max_entries: int) -> tuple[int, CollectionResult]:
        """
        Process a single log file and extract LLM interaction data.
        
        Args:
            file_path: Path to the log file
            start_position: Position to start reading from
            max_entries: Maximum number of entries to process
            
        Returns:
            Tuple of (new position, CollectionResult)
        """
        result = CollectionResult(
            status=CollectionStatus.SUCCESS,
            start_time=datetime.now()
        )
        
        # Detect if the file is gzipped
        is_gzipped = file_path.endswith('.gz')
        
        try:
            if is_gzipped:
                # Can't easily seek in gzipped files, so we read the whole thing
                # and then skip to the position we want
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    lines = f.readlines()
                    lines = lines[start_position:]
                    
                    for i, line in enumerate(lines):
                        if i >= max_entries:
                            break
                            
                        self._process_log_entry(line.strip(), result)
                        
                    new_position = start_position + min(len(lines), max_entries)
            else:
                # For regular files, we can seek to the start position
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.seek(start_position)
                    
                    for _ in range(max_entries):
                        line = f.readline()
                        if not line:
                            break
                            
                        self._process_log_entry(line.strip(), result)
                    
                    new_position = f.tell()
                    
        except Exception as e:
            result.status = CollectionStatus.FAILURE
            result.error = f"Error processing file {file_path}: {str(e)}"
            new_position = start_position  # Don't advance if there was an error
        
        result.end_time = datetime.now()
        return new_position, result
    
    def _process_log_entry(self, line: str, result: CollectionResult) -> None:
        """
        Process a single log entry and extract LLM interaction data.
        
        Args:
            line: Log line to process
            result: CollectionResult to update with processing results
        """
        result.items_processed += 1
        
        try:
            if self.format == "json":
                data = self._extract_json_data(line)
            elif self.format == "regex":
                data = self._extract_regex_data(line)
            else:
                # Unsupported format
                return
                
            if data:
                result.items_collected += 1
                result.data.append(data)
                
                # Notify callbacks if registered
                self._notify_callbacks(data)
                
                # Try to convert to LLM interaction model if possible
                interaction = self._create_interaction(data)
                if interaction:
                    self._notify_callbacks(interaction)
        
        except Exception as e:
            # Log error but continue processing
            print(f"Error processing log entry: {str(e)}")
    
    def _extract_json_data(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Extract LLM interaction data from a JSON log entry.
        
        Args:
            line: JSON log line to process
            
        Returns:
            Extracted data dictionary or None if no valid data
        """
        try:
            # Parse JSON
            data = json.loads(line)
            
            # Navigate JSON path if specified
            if self.json_path:
                parts = self.json_path.split('.')
                for part in parts:
                    if part in data:
                        data = data[part]
                    else:
                        return None
            
            # Verify this is LLM interaction data by checking for required fields
            if self._is_valid_llm_data(data):
                return data
                
            return None
            
        except json.JSONDecodeError:
            return None
    
    def _extract_regex_data(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Extract LLM interaction data using regex pattern.
        
        Args:
            line: Log line to process using regex
            
        Returns:
            Extracted data dictionary or None if no match
        """
        if not self._compiled_regex:
            return None
            
        match = self._compiled_regex.match(line)
        if not match:
            return None
            
        # Convert named groups to dictionary
        data = match.groupdict()
        
        # Try to parse JSON fields if they look like JSON
        for key, value in data.items():
            if value and isinstance(value, str) and value.strip().startswith('{'):
                try:
                    data[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
        
        # Verify this is LLM interaction data
        if self._is_valid_llm_data(data):
            return data
            
        return None
    
    def _is_valid_llm_data(self, data: Dict[str, Any]) -> bool:
        """
        Check if the data represents a valid LLM interaction.
        
        Args:
            data: Data dictionary to validate
            
        Returns:
            Whether the data is valid LLM interaction data
        """
        # Basic validity check - adjust based on your log format
        if isinstance(data, dict):
            # Check for basic request/response structure
            has_request = 'request' in data or 'prompt' in data or 'input' in data
            has_response = 'response' in data or 'completion' in data or 'output' in data
            
            return has_request and has_response
            
        return False
    
    def _create_interaction(self, data: Dict[str, Any]) -> Optional[LLMInteraction]:
        """
        Try to convert raw data to an LLMInteraction object.
        
        Args:
            data: Raw data dictionary from logs
            
        Returns:
            LLMInteraction object or None if conversion not possible
        """
        try:
            # Extract request data
            request_data = data.get('request', {})
            if not isinstance(request_data, dict):
                request_data = {}
                
            # Allow top-level prompt to override request.prompt
            prompt = data.get('prompt') or request_data.get('prompt') or data.get('input') or ''
            
            # Extract response data
            response_data = data.get('response', {})
            if not isinstance(response_data, dict):
                response_data = {}
                
            # Allow top-level completion to override response.completion
            completion = (data.get('completion') or response_data.get('completion') or 
                          data.get('output') or response_data.get('output') or '')
            
            # Extract token info if available
            token_info = None
            tokens_data = data.get('tokens') or response_data.get('tokens')
            if tokens_data:
                input_tokens = tokens_data.get('input', 0) or tokens_data.get('prompt', 0) or 0
                output_tokens = tokens_data.get('output', 0) or tokens_data.get('completion', 0) or 0
                total_tokens = tokens_data.get('total', 0) or 0
                
                token_info = TokenInfo(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    is_estimated=False
                )
            
            # Extract other fields
            model = data.get('model') or request_data.get('model') or response_data.get('model')
            
            # Create request and response
            request_id = data.get('request_id') or request_data.get('id') or ''
            
            request = LLMRequest(
                id=request_id,
                prompt=prompt,
                model=model
            )
            
            response = LLMResponse(
                request_id=request_id,
                completion=completion,
                model=model,
                tokens=token_info
            )
            
            # Create complete interaction
            return LLMInteraction(request=request, response=response)
            
        except Exception as e:
            # Log error but don't fail
            print(f"Error creating LLM interaction: {str(e)}")
            return None
    
    def _monitor_files(self) -> None:
        """
        Background thread method to continuously monitor log files for new data.
        
        This runs until the collector is stopped.
        """
        while self.is_running:
            try:
                # Collect data from log files
                self.collect()
                
                # Check for new log files
                log_files = self._get_log_files()
                for file_path in log_files:
                    if file_path not in self._file_positions and os.path.isfile(file_path):
                        # Start from end for new files
                        self._file_positions[file_path] = os.path.getsize(file_path)
                
                # Wait for next interval
                time.sleep(self.tail_interval)
                
            except Exception as e:
                # Log error but continue
                print(f"Error in log file monitor: {str(e)}")
                time.sleep(self.tail_interval)
