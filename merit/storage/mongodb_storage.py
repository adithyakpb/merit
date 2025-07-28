"""
MongoDB storage implementation for MERIT.

This module provides a comprehensive MongoDB storage backend for persisting MERIT data
such as interactions, metrics, and other serializable objects using MongoDB collections.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import threading
from urllib.parse import quote_plus

try:
    import pymongo
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.errors import (
        ConnectionFailure, ServerSelectionTimeoutError, 
        DuplicateKeyError, BulkWriteError, PyMongoError
    )
    from pymongo.collection import Collection
    from pymongo.database import Database
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

from .persistence import BaseStorage


class MongoDBStorage(BaseStorage):
    """
    MongoDB storage implementation with comprehensive MongoDB functionality.
    
    This storage backend provides full MongoDB capabilities including:
    - Native MongoDB operations (CRUD, aggregation, indexing)
    - Advanced querying with MongoDB query language
    - Collection and database management
    - Transaction support
    - Performance monitoring and optimization
    
    Attributes:
        client: MongoDB client instance
        database: MongoDB database instance
        collection: Default collection for interactions
        config: Storage configuration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MongoDB storage with the given configuration.
        
        Args:
            config: Configuration dictionary with MongoDB settings
                - connection_string: MongoDB connection URI
                - database: Database name
                - collection: Default collection name
                - connection_options: Additional connection options
                - write_concern: Write concern settings
                - read_preference: Read preference
                - authentication: Auth credentials
        """
        super().__init__(config)
        
        if not PYMONGO_AVAILABLE:
            raise ImportError("pymongo is required for MongoDB storage. Install with: pip install pymongo")
        
        # Extract MongoDB specific configuration
        self.connection_string = config.get("connection_string", "mongodb://localhost:27017/")
        self.database_name = config.get("database", "merit_db")
        self.collection_name = config.get("collection", "interactions")
        
        # Connection options
        self.connection_options = config.get("connection_options", {
            "maxPoolSize": 100,
            "minPoolSize": 10,
            "maxIdleTimeMS": 30000,
            "serverSelectionTimeoutMS": 5000,
            "connectTimeoutMS": 10000,
            "socketTimeoutMS": 30000
        })
        
        # Write concern and read preference
        self.write_concern = config.get("write_concern", {"w": "majority", "j": True})
        self.read_preference = config.get("read_preference", "primary")
        
        # Authentication
        auth_config = config.get("authentication", {})
        if auth_config.get("username") and auth_config.get("password"):
            username = quote_plus(auth_config["username"])
            password = quote_plus(auth_config["password"])
            auth_source = auth_config.get("auth_source", "admin")
            
            # Update connection string with auth
            if "://" in self.connection_string:
                protocol, rest = self.connection_string.split("://", 1)
                self.connection_string = f"{protocol}://{username}:{password}@{rest}"
                if "authSource" not in self.connection_string:
                    separator = "&" if "?" in self.connection_string else "?"
                    self.connection_string += f"{separator}authSource={auth_source}"
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize MongoDB connection
        self._initialize_connection()
        
        # Create default indexes
        self._create_default_indexes()
    
    def _initialize_connection(self):
        """Initialize MongoDB connection and validate it."""
        try:
            # Create MongoDB client
            self.client = MongoClient(
                self.connection_string,
                **self.connection_options
            )
            
            # Test connection
            self.client.admin.command('ping')
            
            # Get database and collection references
            self.database: Database = self.client[self.database_name]
            self.collection: Collection = self.database[self.collection_name]
            
            # Configure write concern
            if self.write_concern:
                from pymongo.write_concern import WriteConcern
                wc = WriteConcern(**self.write_concern)
                self.collection = self.collection.with_options(write_concern=wc)
            
            # Configure read preference
            if self.read_preference:
                from pymongo.read_preferences import ReadPreference
                rp = getattr(ReadPreference, self.read_preference.upper(), ReadPreference.PRIMARY)
                self.collection = self.collection.with_options(read_preference=rp)
            
            self.logger.info(f"Connected to MongoDB: {self.database_name}.{self.collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            raise ConnectionError(f"MongoDB connection failed: {e}")
    
    def _create_default_indexes(self):
        """Create default indexes for optimal performance."""
        try:
            # Index on timestamp for time-based queries
            self.collection.create_index([("timestamp", DESCENDING)], background=True)
            
            # Index on model for filtering by model
            self.collection.create_index([("model", ASCENDING)], background=True)
            
            # Compound index for common queries
            self.collection.create_index([
                ("timestamp", DESCENDING),
                ("model", ASCENDING)
            ], background=True)
            
            # Index on request_id for lookup
            self.collection.create_index([("request_id", ASCENDING)], background=True)
            
            self.logger.info("Default indexes created successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to create default indexes: {e}")
    
    # BaseStorage Interface Implementation
    
    def store(self, interactions: List[Dict[str, Any]]) -> bool:
        """
        Store interactions in MongoDB.
        
        Args:
            interactions: List of interaction dictionaries to store
            
        Returns:
            True if storage was successful, False otherwise
        """
        if not interactions:
            return True
            
        try:
            # Prepare documents for insertion
            documents = []
            for interaction in interactions:
                doc = interaction.copy()
                
                # Add timestamp if not present
                if 'timestamp' not in doc:
                    doc['timestamp'] = datetime.now()
                elif isinstance(doc['timestamp'], str):
                    # Convert string timestamp to datetime
                    try:
                        doc['timestamp'] = datetime.fromisoformat(doc['timestamp'].replace('Z', '+00:00'))
                    except ValueError:
                        doc['timestamp'] = datetime.now()
                
                # Add created_at timestamp
                doc['created_at'] = datetime.now()
                
                documents.append(doc)
            
            # Insert documents
            if len(documents) == 1:
                result = self.collection.insert_one(documents[0])
                success = result.acknowledged
            else:
                result = self.collection.insert_many(documents, ordered=False)
                success = result.acknowledged
            
            if success:
                self.logger.info(f"Successfully stored {len(documents)} interactions")
                return True
            else:
                self.logger.error("Failed to store interactions - not acknowledged")
                return False
                
        except Exception as e:
            self.logger.error(f"Error storing interactions in MongoDB: {e}")
            return False
    
    def query(self, 
             query: Optional[Dict[str, Any]] = None, 
             limit: int = 100,
             offset: int = 0,
             sort: Optional[List[Tuple[str, str]]] = None) -> List[Dict[str, Any]]:
        """
        Query interactions from MongoDB.
        
        Args:
            query: MongoDB query conditions
            limit: Maximum number of results to return
            offset: Number of results to skip
            sort: List of (field, direction) tuples for sorting
            
        Returns:
            List of matching interaction dictionaries
        """
        try:
            # Build MongoDB query
            mongo_query = self._build_mongo_query(query) if query else {}
            
            # Build sort specification
            sort_spec = []
            if sort:
                for field, direction in sort:
                    mongo_direction = DESCENDING if direction.lower() == "desc" else ASCENDING
                    sort_spec.append((field, mongo_direction))
            else:
                # Default sort by timestamp descending
                sort_spec = [("timestamp", DESCENDING)]
            
            # Execute query
            cursor = self.collection.find(mongo_query)
            
            if sort_spec:
                cursor = cursor.sort(sort_spec)
            
            cursor = cursor.skip(offset).limit(limit)
            
            # Convert results
            results = []
            for doc in cursor:
                # Remove MongoDB's _id field and convert datetime objects
                doc = self._convert_document_for_output(doc)
                results.append(doc)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error querying interactions from MongoDB: {e}")
            return []
    
    def count(self, query: Optional[Dict[str, Any]] = None, collection_name: Optional[str] = None) -> int:
        """
        Count interactions in MongoDB.
        
        Args:
            query: MongoDB query conditions
            collection_name: Collection name (uses default if None)
            
        Returns:
            Number of matching interactions
        """
        try:
            collection = self._get_collection(collection_name)
            mongo_query = self._build_mongo_query(query) if query else {}
            return collection.count_documents(mongo_query)
            
        except Exception as e:
            self.logger.error(f"Error counting interactions in MongoDB: {e}")
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
            # Calculate cutoff timestamp
            cutoff = datetime.now() - timedelta(days=retention_days)
            
            # Delete old interactions
            result = self.collection.delete_many({"timestamp": {"$lt": cutoff}})
            
            purged = result.deleted_count
            self.logger.info(f"Purged {purged} interactions older than {retention_days} days")
            
            return purged
            
        except Exception as e:
            self.logger.error(f"Error purging interactions from MongoDB: {e}")
            return 0
    
    def close(self):
        """Close the MongoDB connection."""
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
                self.logger.info("MongoDB connection closed")
        except Exception as e:
            self.logger.error(f"Error closing MongoDB connection: {e}")
    
    # Extended MongoDB Operations
    
    def insert_one(self, document: Dict[str, Any], collection_name: Optional[str] = None) -> Optional[str]:
        """
        Insert a single document into MongoDB.
        
        Args:
            document: Document to insert
            collection_name: Collection name (uses default if None)
            
        Returns:
            Inserted document ID as string, None if failed
        """
        try:
            collection = self._get_collection(collection_name)
            result = collection.insert_one(document)
            return str(result.inserted_id) if result.acknowledged else None
        except Exception as e:
            self.logger.error(f"Error inserting document: {e}")
            return None
    
    def insert_many(self, documents: List[Dict[str, Any]], collection_name: Optional[str] = None) -> List[str]:
        """
        Insert multiple documents into MongoDB.
        
        Args:
            documents: List of documents to insert
            collection_name: Collection name (uses default if None)
            
        Returns:
            List of inserted document IDs as strings
        """
        try:
            collection = self._get_collection(collection_name)
            result = collection.insert_many(documents, ordered=False)
            return [str(oid) for oid in result.inserted_ids] if result.acknowledged else []
        except Exception as e:
            self.logger.error(f"Error inserting documents: {e}")
            return []
    
    def find_one(self, query: Dict[str, Any], collection_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Find a single document in MongoDB.
        
        Args:
            query: MongoDB query
            collection_name: Collection name (uses default if None)
            
        Returns:
            Document if found, None otherwise
        """
        try:
            collection = self._get_collection(collection_name)
            doc = collection.find_one(query)
            return self._convert_document_for_output(doc) if doc else None
        except Exception as e:
            self.logger.error(f"Error finding document: {e}")
            return None
    
    def find(self, query: Dict[str, Any], limit: int = 100, skip: int = 0, 
             sort: Optional[List[Tuple[str, int]]] = None, 
             collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find multiple documents in MongoDB.
        
        Args:
            query: MongoDB query
            limit: Maximum number of results
            skip: Number of results to skip
            sort: Sort specification
            collection_name: Collection name (uses default if None)
            
        Returns:
            List of matching documents
        """
        try:
            collection = self._get_collection(collection_name)
            cursor = collection.find(query)
            
            if sort:
                cursor = cursor.sort(sort)
            
            cursor = cursor.skip(skip)
            if limit is not None and limit > 0:
                cursor = cursor.limit(limit)

            return [self._convert_document_for_output(doc) for doc in cursor]
        except Exception as e:
            self.logger.error(f"Error finding documents: {e}")
            return []
    
    def update_one(self, query: Dict[str, Any], update: Dict[str, Any], 
                   upsert: bool = False, collection_name: Optional[str] = None) -> bool:
        """
        Update a single document in MongoDB.
        
        Args:
            query: Query to match document
            update: Update operations
            upsert: Whether to insert if not found
            collection_name: Collection name (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self._get_collection(collection_name)
            result = collection.update_one(query, update, upsert=upsert)
            return result.acknowledged
        except Exception as e:
            self.logger.error(f"Error updating document: {e}")
            return False
    
    def update_many(self, query: Dict[str, Any], update: Dict[str, Any], 
                    upsert: bool = False, collection_name: Optional[str] = None) -> int:
        """
        Update multiple documents in MongoDB.
        
        Args:
            query: Query to match documents
            update: Update operations
            upsert: Whether to insert if not found
            collection_name: Collection name (uses default if None)
            
        Returns:
            Number of documents updated
        """
        try:
            collection = self._get_collection(collection_name)
            result = collection.update_many(query, update, upsert=upsert)
            return result.modified_count if result.acknowledged else 0
        except Exception as e:
            self.logger.error(f"Error updating documents: {e}")
            return 0
    
    def delete_one(self, query: Dict[str, Any], collection_name: Optional[str] = None) -> bool:
        """
        Delete a single document from MongoDB.
        
        Args:
            query: Query to match document
            collection_name: Collection name (uses default if None)
            
        Returns:
            True if a document was deleted, False otherwise
        """
        try:
            collection = self._get_collection(collection_name)
            result = collection.delete_one(query)
            return result.deleted_count > 0
        except Exception as e:
            self.logger.error(f"Error deleting document: {e}")
            return False
    
    def delete_many(self, query: Dict[str, Any], collection_name: Optional[str] = None) -> int:
        """
        Delete multiple documents from MongoDB.
        
        Args:
            query: Query to match documents
            collection_name: Collection name (uses default if None)
            
        Returns:
            Number of documents deleted
        """
        try:
            collection = self._get_collection(collection_name)
            result = collection.delete_many(query)
            return result.deleted_count
        except Exception as e:
            self.logger.error(f"Error deleting documents: {e}")
            return 0
    
    def aggregate(self, pipeline: List[Dict[str, Any]], collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Execute an aggregation pipeline.
        
        Args:
            pipeline: Aggregation pipeline stages
            collection_name: Collection name (uses default if None)
            
        Returns:
            List of aggregation results
        """
        try:
            collection = self._get_collection(collection_name)
            cursor = collection.aggregate(pipeline)
            # Preserve _id in aggregation results as it's often meaningful
            return [self._convert_document_for_output(doc, preserve_id=True) for doc in cursor]
        except Exception as e:
            self.logger.error(f"Error executing aggregation: {e}")
            return []
    
    def bulk_write(self, operations: List[Any], collection_name: Optional[str] = None) -> Dict[str, int]:
        """
        Execute bulk write operations.
        
        Args:
            operations: List of bulk write operations
            collection_name: Collection name (uses default if None)
            
        Returns:
            Dictionary with operation counts
        """
        try:
            collection = self._get_collection(collection_name)
            result = collection.bulk_write(operations, ordered=False)
            
            return {
                "inserted": result.inserted_count,
                "matched": result.matched_count,
                "modified": result.modified_count,
                "deleted": result.deleted_count,
                "upserted": result.upserted_count
            }
        except Exception as e:
            self.logger.error(f"Error executing bulk write: {e}")
            return {"inserted": 0, "matched": 0, "modified": 0, "deleted": 0, "upserted": 0}
    
    # Index Management
    
    def create_index(self, keys: Union[str, List[Tuple[str, int]]], 
                     collection_name: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Create an index on the collection.
        
        Args:
            keys: Index specification
            collection_name: Collection name (uses default if None)
            **kwargs: Additional index options
            
        Returns:
            Index name if successful, None otherwise
        """
        try:
            collection = self._get_collection(collection_name)
            return collection.create_index(keys, **kwargs)
        except Exception as e:
            self.logger.error(f"Error creating index: {e}")
            return None
    
    def drop_index(self, index_name: str, collection_name: Optional[str] = None) -> bool:
        """
        Drop an index from the collection.
        
        Args:
            index_name: Name of the index to drop
            collection_name: Collection name (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self._get_collection(collection_name)
            collection.drop_index(index_name)
            return True
        except Exception as e:
            self.logger.error(f"Error dropping index: {e}")
            return False
    
    def list_indexes(self, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all indexes on the collection.
        
        Args:
            collection_name: Collection name (uses default if None)
            
        Returns:
            List of index information
        """
        try:
            collection = self._get_collection(collection_name)
            return list(collection.list_indexes())
        except Exception as e:
            self.logger.error(f"Error listing indexes: {e}")
            return []
    
    # Collection Management
    
    def create_collection(self, collection_name: str, **kwargs) -> bool:
        """
        Create a new collection.
        
        Args:
            collection_name: Name of the collection to create
            **kwargs: Additional collection options
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.database.create_collection(collection_name, **kwargs)
            return True
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            return False
    
    def drop_collection(self, collection_name: str) -> bool:
        """
        Drop a collection.
        
        Args:
            collection_name: Name of the collection to drop
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.database.drop_collection(collection_name)
            return True
        except Exception as e:
            self.logger.error(f"Error dropping collection: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """
        List all collections in the database.
        
        Returns:
            List of collection names
        """
        try:
            return self.database.list_collection_names()
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            return []
    
    # Database Administration
    
    def database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Database statistics
        """
        try:
            stats = self.database.command("dbStats")
            return self._convert_document_for_output(stats)
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}
    
    def collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Args:
            collection_name: Collection name (uses default if None)
            
        Returns:
            Collection statistics
        """
        try:
            coll_name = collection_name or self.collection_name
            stats = self.database.command("collStats", coll_name)
            return self._convert_document_for_output(stats)
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def explain_query(self, query: Dict[str, Any], collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get query execution plan.
        
        Args:
            query: Query to explain
            collection_name: Collection name (uses default if None)
            
        Returns:
            Query execution plan
        """
        try:
            collection = self._get_collection(collection_name)
            plan = collection.find(query).explain()
            return self._convert_document_for_output(plan)
        except Exception as e:
            self.logger.error(f"Error explaining query: {e}")
            return {}
    
    # Helper Methods
    
    def _get_collection(self, collection_name: Optional[str] = None) -> "Collection":
        """Get collection reference."""
        if collection_name:
            return self.database[collection_name]
        return self.collection
    
    def _build_mongo_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert generic query to MongoDB query.
        
        Args:
            query: Generic query conditions
            
        Returns:
            MongoDB query
        """
        mongo_query = {}
        
        for key, value in query.items():
            if isinstance(value, dict):
                # Handle operators
                mongo_operators = {}
                for op, op_value in value.items():
                    if op == "$gte":
                        mongo_operators["$gte"] = self._convert_value_for_query(op_value)
                    elif op == "$gt":
                        mongo_operators["$gt"] = self._convert_value_for_query(op_value)
                    elif op == "$lte":
                        mongo_operators["$lte"] = self._convert_value_for_query(op_value)
                    elif op == "$lt":
                        mongo_operators["$lt"] = self._convert_value_for_query(op_value)
                    elif op == "$eq":
                        mongo_operators["$eq"] = self._convert_value_for_query(op_value)
                    elif op == "$ne":
                        mongo_operators["$ne"] = self._convert_value_for_query(op_value)
                    elif op == "$in":
                        mongo_operators["$in"] = [self._convert_value_for_query(v) for v in op_value]
                    elif op == "$nin":
                        mongo_operators["$nin"] = [self._convert_value_for_query(v) for v in op_value]
                    else:
                        # Pass through other MongoDB operators
                        mongo_operators[op] = op_value
                
                mongo_query[key] = mongo_operators
            else:
                # Direct equality
                mongo_query[key] = self._convert_value_for_query(value)
        
        return mongo_query
    
    def _convert_value_for_query(self, value: Any) -> Any:
        """Convert value for MongoDB query."""
        if isinstance(value, str):
            # Try to convert ISO timestamp strings to datetime
            try:
                if 'T' in value or value.endswith('Z'):
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                pass
        
        return value
    
    def _convert_document_for_output(self, doc: Dict[str, Any], preserve_id: bool = False) -> Dict[str, Any]:
        """Convert MongoDB document for output."""
        if not doc:
            return doc
            
        # Remove MongoDB's _id field unless explicitly preserving it
        if '_id' in doc and not preserve_id:
            del doc['_id']
        
        # Convert datetime objects to ISO strings
        for key, value in doc.items():
            if isinstance(value, datetime):
                doc[key] = value.isoformat()
            elif isinstance(value, dict):
                doc[key] = self._convert_document_for_output(value)
            elif isinstance(value, list):
                doc[key] = [
                    self._convert_document_for_output(item) if isinstance(item, dict) 
                    else item.isoformat() if isinstance(item, datetime) 
                    else item 
                    for item in value
                ]
        
        return doc
