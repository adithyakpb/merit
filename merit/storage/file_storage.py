"""
File-based storage implementation for MERIT.

This module provides a file-based storage backend for persisting MERIT data
such as interactions, metrics, and other serializable objects using JSON files.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import threading

from .persistence import BaseStorage


class FileStorage(BaseStorage):
    """
    File-based storage implementation using JSON files.
    
    This storage backend persists data to JSON files, with options for
    different file organization strategies.
    
    Attributes:
        root_path: Root directory for stored files
        file_organization: Strategy for organizing files ('flat', 'date', 'id')
        max_file_size: Maximum file size in bytes before starting a new file
        compression: Whether to use compression for stored files
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize file storage with the given configuration.
        
        Args:
            config: Configuration dictionary with storage settings
        """
        super().__init__(config)
        
        # Extract file storage specific configuration
        self.root_path = config.get("root_path", "./merit_data")
        self.file_organization = config.get("file_organization", "date")
        self.max_file_size = config.get("max_file_size", 10 * 1024 * 1024)  # 10MB default
        self.compression = config.get("compression", False)
        
        # Make sure the root directory exists
        os.makedirs(self.root_path, exist_ok=True)
        
        # Prepare the interaction directory
        self.interaction_dir = os.path.join(self.root_path, "interactions")
        os.makedirs(self.interaction_dir, exist_ok=True)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # File handle caching
        self._current_file = None
        self._current_file_size = 0
        
    def store(self, interactions: List[Dict[str, Any]]) -> bool:
        """
        Store interactions to the file storage.
        
        Args:
            interactions: List of interaction dictionaries to store
            
        Returns:
            True if storage was successful, False otherwise
        """
        if not interactions:
            return True
            
        try:
            # Determine file path based on organization strategy
            file_path = self._get_file_path()
            
            # Write the interactions to the file
            with self._lock:
                with open(file_path, 'a') as f:
                    for interaction in interactions:
                        # Add a timestamp if not present
                        if 'timestamp' not in interaction:
                            interaction['timestamp'] = datetime.now().isoformat()
                        
                        # Write as JSON line
                        json_line = json.dumps(interaction) + "\n"
                        f.write(json_line)
                        
                        # Update file size tracking
                        self._current_file_size += len(json_line)
                
                # Reset file tracking if we've exceeded the max file size
                if self._current_file_size > self.max_file_size:
                    self._current_file = None
                    self._current_file_size = 0
            
            return True
        except Exception as e:
            self.logger.error(f"Error storing interactions to file: {e}")
            return False

    def query(self, 
             query: Optional[Dict[str, Any]] = None, 
             limit: int = 100,
             offset: int = 0,
             sort: Optional[List[Tuple[str, str]]] = None) -> List[Dict[str, Any]]:
        """
        Query interactions from file storage.
        
        This is a simple implementation that scans files. In production,
        you'd want a more sophisticated approach for large datasets.
        
        Args:
            query: Query conditions (simple key-value matching)
            limit: Maximum number of results to return
            offset: Number of results to skip
            sort: List of (field, direction) tuples for sorting
            
        Returns:
            List of matching interaction dictionaries
        """
        results = []
        
        try:
            # Get all files in reverse chronological order (newest first)
            files = self._get_all_files()
            files.sort(reverse=True)
            
            # Simple in-memory matching (not efficient for large datasets)
            skipped = 0
            for file_path in files:
                if len(results) >= limit:
                    break
                    
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            interaction = json.loads(line.strip())
                            
                            # Apply filtering
                            if query and not self._matches_query(interaction, query):
                                continue
                                
                            # Apply offset
                            if skipped < offset:
                                skipped += 1
                                continue
                                
                            # Add to results
                            results.append(interaction)
                            
                            # Check limit
                            if len(results) >= limit:
                                break
                        except json.JSONDecodeError:
                            continue
            
            # Apply sorting if specified
            if sort:
                for field, direction in reversed(sort):
                    reverse = direction.lower() == "desc"
                    results.sort(key=lambda x: x.get(field, ""), reverse=reverse)
            
            return results
        except Exception as e:
            self.logger.error(f"Error querying file storage: {e}")
            return []
    
    def get_interactions(self, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get interactions from a specific time range.
        
        Args:
            start_time: Start of the time range (inclusive)
            end_time: End of the time range (inclusive)
            limit: Maximum number of results to return
            
        Returns:
            List of interaction dictionaries
        """
        query = {}
        
        if start_time or end_time:
            query["timestamp"] = {}
            
            if start_time:
                query["timestamp"]["$gte"] = start_time.isoformat()
                
            if end_time:
                query["timestamp"]["$lte"] = end_time.isoformat()
        
        return self.query(query=query, limit=limit)
    
    def delete_interactions(self, query: Dict[str, Any]) -> int:
        """
        Delete interactions matching the query.
        
        This is a simple implementation that recreates files. In production,
        you'd want a more sophisticated approach.
        
        Args:
            query: Query condition for interactions to delete
            
        Returns:
            Number of deleted interactions
        """
        deleted_count = 0
        
        try:
            # Process each file
            files = self._get_all_files()
            
            for file_path in files:
                temp_path = file_path + ".temp"
                deleted_in_file = 0
                
                # Read the file and write non-matching interactions to temp file
                with open(file_path, 'r') as input_file, open(temp_path, 'w') as output_file:
                    for line in input_file:
                        try:
                            interaction = json.loads(line.strip())
                            
                            # If it doesn't match the query, keep it
                            if not self._matches_query(interaction, query):
                                output_file.write(line)
                            else:
                                deleted_in_file += 1
                                deleted_count += 1
                        except json.JSONDecodeError:
                            # Keep lines we can't parse
                            output_file.write(line)
                
                # If we deleted anything, replace the original file
                if deleted_in_file > 0:
                    os.replace(temp_path, file_path)
                else:
                    # No deletions, just remove the temp file
                    os.remove(temp_path)
        except Exception as e:
            self.logger.error(f"Error deleting interactions: {e}")
        
        return deleted_count
    
    def prune(self, days: int = 30) -> int:
        """
        Remove interactions older than the specified number of days.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of deleted interactions
        """
        if days <= 0:
            return 0
            
        # Calculate cutoff date
        cutoff = datetime.now() - timedelta(days=days)
        
        # Delete interactions older than cutoff
        query = {
            "timestamp": {
                "$lt": cutoff.isoformat()
            }
        }
        
        return self.delete_interactions(query)
    
    def _get_file_path(self) -> str:
        """
        Get the file path for storing new interactions based on the configuration.
        
        Returns:
            Path to the file for writing
        """
        # If we have a current file that's not full, use it
        if self._current_file and self._current_file_size < self.max_file_size:
            return self._current_file
            
        # Otherwise, create a new file
        now = datetime.now()
        
        if self.file_organization == "flat":
            # Simple flat organization with timestamp
            filename = f"interactions_{int(time.time())}.jsonl"
            path = os.path.join(self.interaction_dir, filename)
        
        elif self.file_organization == "date":
            # Organize by year/month/day
            year_dir = os.path.join(self.interaction_dir, str(now.year))
            month_dir = os.path.join(year_dir, f"{now.month:02d}")
            day_dir = os.path.join(month_dir, f"{now.day:02d}")
            
            # Create directories
            os.makedirs(day_dir, exist_ok=True)
            
            # Create file
            filename = f"interactions_{now.hour:02d}_{now.minute:02d}_{int(time.time())}.jsonl"
            path = os.path.join(day_dir, filename)
            
        else:  # default or "id" organization
            # Use a uuid-based organization
            import uuid
            filename = f"interactions_{uuid.uuid4()}.jsonl"
            path = os.path.join(self.interaction_dir, filename)
        
        # Update tracking
        self._current_file = path
        self._current_file_size = 0
        
        return path
    
    def _get_all_files(self) -> List[str]:
        """
        Get all interaction files in the storage.
        
        Returns:
            List of file paths
        """
        files = []
        
        # Walk the directory structure
        for root, _, filenames in os.walk(self.interaction_dir):
            for filename in filenames:
                if filename.endswith('.jsonl') and not filename.endswith('.temp'):
                    files.append(os.path.join(root, filename))
        
        return files
    
    def _matches_query(self, interaction: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """
        Check if an interaction matches a query.
        
        This is a simple implementation supporting basic operations.
        
        Args:
            interaction: The interaction to check
            query: The query conditions
            
        Returns:
            True if the interaction matches the query
        """
        for key, value in query.items():
            # Handle nested keys
            parts = key.split('.')
            curr = interaction
            
            # Navigate to the nested value
            for part in parts:
                if isinstance(curr, dict) and part in curr:
                    curr = curr[part]
                else:
                    return False
            
            # Check the value
            if isinstance(value, dict):
                # Handle operators
                for op, op_value in value.items():
                    if op == "$eq":
                        if curr != op_value:
                            return False
                    elif op == "$ne":
                        if curr == op_value:
                            return False
                    elif op == "$gt":
                        if not curr > op_value:
                            return False
                    elif op == "$gte":
                        if not curr >= op_value:
                            return False
                    elif op == "$lt":
                        if not curr < op_value:
                            return False
                    elif op == "$lte":
                        if not curr <= op_value:
                            return False
                    elif op == "$in":
                        if curr not in op_value:
                            return False
                    elif op == "$nin":
                        if curr in op_value:
                            return False
            else:
                # Direct equality
                if curr != value:
                    return False
        
        return True
