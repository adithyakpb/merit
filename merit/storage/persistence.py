"""
Storage persistence interface and implementations for MERIT.

This module provides the base storage interface and concrete implementations
for persisting MERIT data such as monitoring interactions and metrics.
"""

import os
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

class BaseStorage:
    """
    Base class for MERIT storage backends.
    
    This defines the common interface that all storage implementations must follow.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the storage backend with configuration.
        
        Args:
            config: Configuration dictionary for the storage backend
        """
        self.config = config
        self.logger = None  # Will be set later
        
        try:
            from ...core.logging import get_logger
            self.logger = get_logger(__name__)
        except ImportError:
            import logging
            self.logger = logging.getLogger(__name__)
    
    def store(self, interactions: List[Dict[str, Any]]) -> bool:
        """
        Store interactions in the storage backend.
        
        Args:
            interactions: List of interaction dictionaries to store
            
        Returns:
            True if the operation was successful, False otherwise
        """
        raise NotImplementedError
        
    def retrieve(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve interactions from storage.
        
        Args:
            limit: Maximum number of interactions to retrieve
            offset: Number of interactions to skip
            
        Returns:
            List of interaction dictionaries
        """
        return self.query(limit=limit, offset=offset)
    
    def query(self, 
             query: Optional[Dict[str, Any]] = None, 
             limit: int = 100,
             offset: int = 0,
             sort: Optional[List[Tuple[str, str]]] = None) -> List[Dict[str, Any]]:
        """
        Query interactions from storage.
        
        Args:
            query: Query conditions
            limit: Maximum number of results to return
            offset: Number of results to skip
            sort: List of (field, direction) tuples for sorting
            
        Returns:
            List of matching interaction dictionaries
        """
        raise NotImplementedError
    
    def count(self, query: Optional[Dict[str, Any]] = None) -> int:
        """
        Count interactions in storage.
        
        Args:
            query: Query conditions
            
        Returns:
            Number of matching interactions
        """
        raise NotImplementedError
    
    def purge(self, retention_days: int = 30) -> int:
        """
        Remove interactions older than the specified retention period.
        
        Args:
            retention_days: Number of days to retain data for
            
        Returns:
            Number of interactions purged
        """
        raise NotImplementedError
    
    def close(self):
        """Close the storage backend and release resources."""
        pass


class SQLiteStorage(BaseStorage):
    """
    SQLite storage implementation for MERIT.
    
    This uses a SQLite database to store interactions with reasonable performance
    and no external dependencies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SQLite storage with configuration.
        
        Args:
            config: Configuration dictionary with the following options:
                - db_path: Path to the SQLite database file
                - table_prefix: Prefix for table names (default: "merit_")
                - pragma: Additional PRAGMA statements to execute
        """
        super().__init__(config)
        
        # Get configuration
        if isinstance(config, str):
            self.db_path = config
            self.table_prefix = "merit_"
            self.pragma = {}
        else:
            self.db_path = config.get("db_path", "merit.db")
            self.table_prefix = config.get("table_prefix", "merit_")
            self.pragma = config.get("pragma", {})
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        
        # Thread locals for connection handling
        self._local = threading.local()
        
        # Initialize the database
        self._initialize_db()
    
    def _get_connection(self):
        """
        Get a thread-local SQLite connection.
        
        Returns:
            sqlite3.Connection: SQLite connection object
        """
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.row_factory = sqlite3.Row
            
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            
            # Apply additional PRAGMA statements
            for key, value in self.pragma.items():
                self._local.connection.execute(f"PRAGMA {key} = {value}")
        
        return self._local.connection
    
    def _initialize_db(self):
        """Initialize the database schema."""
        conn = self._get_connection()
        
        # Create interactions table
        conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {self.table_prefix}interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            data TEXT NOT NULL,
            model TEXT,
            request_id TEXT,
            status TEXT,
            error TEXT,
            latency REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create index on timestamp for faster queries
        conn.execute(f"""
        CREATE INDEX IF NOT EXISTS {self.table_prefix}idx_interactions_timestamp
        ON {self.table_prefix}interactions (timestamp)
        """)
        
        # Create index on model for filtering by model
        conn.execute(f"""
        CREATE INDEX IF NOT EXISTS {self.table_prefix}idx_interactions_model
        ON {self.table_prefix}interactions (model)
        """)
        
        conn.commit()
    
    def store(self, interactions: List[Dict[str, Any]]) -> bool:
        """
        Store interactions in the SQLite database.
        
        Args:
            interactions: List of interaction dictionaries to store
            
        Returns:
            True if the operation was successful, False otherwise
        """
        if not interactions:
            return True
            
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            for interaction in interactions:
                # Extract common fields for indexing
                timestamp = interaction.get("timestamp")
                if not timestamp:
                    timestamp = datetime.now().isoformat()
                
                model = None
                request_id = None
                status = None
                error = None
                latency = None
                input_tokens = None
                output_tokens = None
                total_tokens = None
                
                # Try to extract fields from the interaction structure
                try:
                    model = interaction.get("model")
                    
                    # Extract request ID
                    if "request" in interaction and isinstance(interaction["request"], dict):
                        request_id = interaction["request"].get("id")
                    
                    # Extract status and error
                    if "response" in interaction and isinstance(interaction["response"], dict):
                        status = interaction["response"].get("status")
                        error = interaction["response"].get("error")
                        latency = interaction["response"].get("latency")
                    
                    # Extract token information
                    if "tokens" in interaction and isinstance(interaction["tokens"], dict):
                        input_tokens = interaction["tokens"].get("input_tokens")
                        output_tokens = interaction["tokens"].get("output_tokens")
                        total_tokens = interaction["tokens"].get("total_tokens")
                except:
                    # If we can't extract the fields, just store the data as JSON
                    pass
                
                # Store the interaction
                cursor.execute(
                    f"""
                    INSERT INTO {self.table_prefix}interactions
                    (timestamp, data, model, request_id, status, error, latency,
                    input_tokens, output_tokens, total_tokens)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        json.dumps(interaction),
                        model,
                        request_id,
                        status,
                        error,
                        latency,
                        input_tokens,
                        output_tokens,
                        total_tokens
                    )
                )
            
            conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing interactions in SQLite: {e}")
            
            if conn:
                conn.rollback()
                
            return False
    
    def query(self, 
             query: Optional[Dict[str, Any]] = None, 
             limit: int = 100,
             offset: int = 0,
             sort: Optional[List[Tuple[str, str]]] = None) -> List[Dict[str, Any]]:
        """
        Query interactions from the SQLite database.
        
        Args:
            query: Query conditions
            limit: Maximum number of results to return
            offset: Number of results to skip
            sort: List of (field, direction) tuples for sorting
            
        Returns:
            List of matching interaction dictionaries
        """
        try:
            conn = self._get_connection()
            
            # Build the SQL query
            sql = f"SELECT data FROM {self.table_prefix}interactions"
            params = []
            
            # Add WHERE clause if query is provided
            if query:
                where_clauses = []
                
                for key, value in query.items():
                    if key == "timestamp":
                        if isinstance(value, dict):
                            # Handle timestamp comparisons
                            for op, op_value in value.items():
                                if op == "$gte":
                                    where_clauses.append("timestamp >= ?")
                                    params.append(op_value)
                                elif op == "$gt":
                                    where_clauses.append("timestamp > ?")
                                    params.append(op_value)
                                elif op == "$lte":
                                    where_clauses.append("timestamp <= ?")
                                    params.append(op_value)
                                elif op == "$lt":
                                    where_clauses.append("timestamp < ?")
                                    params.append(op_value)
                        else:
                            # Exact match
                            where_clauses.append("timestamp = ?")
                            params.append(value)
                    elif key == "model":
                        where_clauses.append("model = ?")
                        params.append(value)
                    elif key == "status":
                        where_clauses.append("status = ?")
                        params.append(value)
                
                if where_clauses:
                    sql += " WHERE " + " AND ".join(where_clauses)
            
            # Add ORDER BY clause
            if sort:
                order_clauses = []
                for field, direction in sort:
                    # Map common fields to their database column names
                    if field == "timestamp":
                        order_clauses.append(f"timestamp {direction}")
                
                if order_clauses:
                    sql += " ORDER BY " + ", ".join(order_clauses)
            else:
                # Default sort by timestamp descending (newest first)
                sql += " ORDER BY timestamp DESC"
            
            # Add LIMIT clause
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Execute the query
            cursor = conn.execute(sql, params)
            
            # Process the results
            results = []
            for row in cursor.fetchall():
                interaction = json.loads(row[0])
                results.append(interaction)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error querying interactions from SQLite: {e}")
            return []
    
    def count(self, query: Optional[Dict[str, Any]] = None) -> int:
        """
        Count interactions in the SQLite database.
        
        Args:
            query: Query conditions
            
        Returns:
            Number of matching interactions
        """
        try:
            conn = self._get_connection()
            
            # Build the SQL query
            sql = f"SELECT COUNT(*) FROM {self.table_prefix}interactions"
            params = []
            
            # Add WHERE clause if query is provided
            if query:
                where_clauses = []
                
                for key, value in query.items():
                    if key == "timestamp":
                        if isinstance(value, dict):
                            # Handle timestamp comparisons
                            for op, op_value in value.items():
                                if op == "$gte":
                                    where_clauses.append("timestamp >= ?")
                                    params.append(op_value)
                                elif op == "$gt":
                                    where_clauses.append("timestamp > ?")
                                    params.append(op_value)
                                elif op == "$lte":
                                    where_clauses.append("timestamp <= ?")
                                    params.append(op_value)
                                elif op == "$lt":
                                    where_clauses.append("timestamp < ?")
                                    params.append(op_value)
                        else:
                            # Exact match
                            where_clauses.append("timestamp = ?")
                            params.append(value)
                    elif key == "model":
                        where_clauses.append("model = ?")
                        params.append(value)
                    elif key == "status":
                        where_clauses.append("status = ?")
                        params.append(value)
                
                if where_clauses:
                    sql += " WHERE " + " AND ".join(where_clauses)
            
            # Execute the query
            cursor = conn.execute(sql, params)
            count = cursor.fetchone()[0]
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error counting interactions in SQLite: {e}")
            return 0
    
    def purge(self, retention_days: int = 30) -> int:
        """
        Remove interactions older than the specified retention period.
        
        Args:
            retention_days: Number of days to retain data for
            
        Returns:
            Number of interactions purged
        """
        try:
            conn = self._get_connection()
            
            # Calculate the cutoff timestamp
            cutoff = datetime.now() - timedelta(days=retention_days)
            cutoff_timestamp = cutoff.isoformat()
            
            # Delete old interactions
            cursor = conn.execute(
                f"DELETE FROM {self.table_prefix}interactions WHERE timestamp < ?",
                (cutoff_timestamp,)
            )
            
            purged = cursor.rowcount
            conn.commit()
            
            return purged
            
        except Exception as e:
            self.logger.error(f"Error purging interactions from SQLite: {e}")
            
            if conn:
                conn.rollback()
                
            return 0
    
    def close(self):
        """Close the database connection."""
        if hasattr(self._local, "connection") and self._local.connection is not None:
            self._local.connection.close()
            self._local.connection = None


class DatabaseFactory:
    """
    Factory for creating storage instances.
    
    This provides a convenient way to create storage instances based on
    configuration.
    """
    
    @staticmethod
    def create(config: Dict[str, Any]) -> BaseStorage:
        """
        Create a storage instance based on configuration.
        
        Args:
            config: Configuration dictionary with storage settings
                - type: Storage type ('sqlite', 'file', etc.)
                - other type-specific settings
                
        Returns:
            Storage instance
        """
        storage_type = config.get("type", "sqlite").lower()
        
        if storage_type == "sqlite":
            return SQLiteStorage(config.get("sqlite", {}))
        elif storage_type == "file":
            from .file_storage import FileStorage
            return FileStorage(config.get("file", {}))
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
