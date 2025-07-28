"""
Merit Knowledge Base

This module provides the knowledge base implementation for the Merit system.
"""
from datetime import datetime, timezone
import numpy as np
from typing import List, Dict, Any, Optional, Union, Sequence, Tuple
import uuid




from ..api.base import BaseAPIClient
from ..core.models import Document
from .prompts import TOPIC_GENERATION_PROMPT
from ..core.utils import detect_language, cosine_similarity, batch_iterator
from ..core.logging import get_logger
from ..storage import MongoDBStorage, DatabaseFactory

logger = get_logger(__name__)

# Constants
DEFAULT_BATCH_SIZE = 32
DEFAULT_MIN_TOPIC_SIZE = 3
DEFAULT_LANGUAGE_DETECTION_SAMPLE_SIZE = 10
DEFAULT_LANGUAGE_DETECTION_MAX_TEXT_LENGTH = 300

#TODO enable connection to milvus KB, Mongo db, sqlite db
#TODO create Document objects for each document
#TODO constructor creates documents from any file (.csv, .pdf, .txt, .docx) 
#TODO enable keyword search, embedding search as and when required. also have functions checking if these are available, so that we do not run into errors, and the user does not set search options when they are not available 
#NOTE if not connecting to a vectorstore we can have options to create an embedding store and then do embedding search + warnings telling that embedding search will actually create embeddings of all documents in the knowledge base. if connecting to a recognised vectorstore, we can avoid this, as the functions will be there to query/search.
#TODO define a search/query function

class KnowledgeBase:
    """
    A knowledge base for the Merit RAG system.
    
    This class provides functionality for creating and managing a knowledge base,
    including embedding documents, finding topics, and searching for relevant documents.
    """
    
    @classmethod
    def from_knowledge_bases(cls, knowledge_bases: List["KnowledgeBase"], client: BaseAPIClient = None) -> "KnowledgeBase":
        """
        Create a combined knowledge base from a list of knowledge bases.
        
        Args:
            knowledge_bases: A list of knowledge bases to combine.
            client: The API client to use for the combined knowledge base. If None, uses the client from the first knowledge base.
            
        Returns:
            KnowledgeBase: A new knowledge base containing all documents from the input knowledge bases.
            
        Raises:
            ValueError: If the list of knowledge bases is empty.
        """
        if not knowledge_bases:
            raise ValueError("Cannot create a knowledge base from an empty list of knowledge bases")
        
        # Use the client from the first knowledge base if not provided
        if client is None:
            client = knowledge_bases[0]._client
        
        # Collect all documents from all knowledge bases
        all_documents = []
        for kb in knowledge_bases:
            all_documents.extend(kb.documents)
        
        # Create document dictionaries
        document_dicts = []
        for doc in all_documents:
            document_dict = {
                "content": doc.content,
                "metadata": doc.metadata,
                "id": doc.id
            }
            document_dicts.append(document_dict)
        
        # Create a new knowledge base with all documents
        return cls(data=document_dicts, client=client)
    
    def __init__(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        client: BaseAPIClient = None,
        columns: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        min_topic_size: Optional[int] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        storage_config: Optional[Dict[str, Any]] = None,
        kb_id: Optional[str] = None,
    ):
        """
        Initialize the knowledge base.
        
        Args:
            data: The data to create the knowledge base from, either a list of dictionaries or a list of Document objects.
                 Can be None if loading from storage.
            client: The API client to use for embeddings and text generation.
            columns: The columns to use from the data. If None, all columns are used.
            seed: The random seed to use.
            min_topic_size: The minimum number of documents to form a topic.
            batch_size: The batch size to use for embeddings.
            storage_config: Configuration for persistent storage. If provided, enables storage integration.
            kb_id: Knowledge base identifier for storage. If None, generates a unique ID.
        """
        # Set up random number generator
        self._rng = np.random.default_rng(seed=seed)
        
        # Store parameters
        self._client = client
        self._batch_size = batch_size
        self._min_topic_size = min_topic_size or DEFAULT_MIN_TOPIC_SIZE
        self._kb_id = kb_id or f"kb_{int(datetime.now(timezone.utc).timestamp())}"
        
        # Initialize storage
        self._storage = None
        if storage_config:
            self._storage = self._initialize_storage(storage_config)
            logger.info(f"Initialized storage backend: {storage_config.get('type', 'unknown')}")
        
        # Load or create documents
        if data is not None:
            # Create documents from data
            self._documents = self._create_documents(data, columns)
            
            if len(self._documents) == 0:
                raise ValueError("Cannot create a knowledge base with empty documents")
        
        elif self._storage:
            # Load documents from storage
            self._documents = self._load_documents_from_storage()
            
            if len(self._documents) == 0:
                raise ValueError("No documents found in storage and no data provided")
        
        else:
            raise ValueError("Either data or storage_config with existing data must be provided")
        
        # Create document index
        self._document_index = {doc.id: doc for doc in self._documents}
        
        # Initialize caches
        self._embeddings_cache = None
        self._topics_cache = None
        self._index_cache = None
        self._reduced_embeddings_cache = None
        
        # Detect language
        self._language = self._detect_language()
        
        # Save to storage if configured and we have new data
        if self._storage and data is not None:
            self._save_documents_to_storage()
        
        logger.info(f"Created knowledge base '{self._kb_id}' with {len(self._documents)} documents in language '{self._language}'")
    
    def _create_documents(self, data: List[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> List[Document]:
        """
        Create documents from the data.
        
        Args:
            data: The data to create documents from.
            columns: The columns to use from the data. If None, all columns are used.
            
        Returns:
            List[Document]: The created documents.
        """
        documents = []
        for item in data:
            content = "\n".join(f"{k}: {v}" for k, v in item.items() if (columns is None or k in columns) and v is not None)
            metadata = {k: v for k, v in item.items() if (columns is None or k in columns)}
            doc = Document(content=content, metadata=metadata)
            documents.append(doc)
        return documents
    
    def _initialize_storage(self, storage_config: Dict[str, Any]):
        """
        Initialize the storage backend.
        
        Args:
            storage_config: Storage configuration dictionary
            
        Returns:
            Storage instance
        """
        try:
            if storage_config.get("type") == "mongodb":
                # Use MongoDB storage directly
                mongodb_config = storage_config.get("mongodb", {})
                # Set default collections for knowledge base
                if "documents_collection" not in mongodb_config:
                    mongodb_config["collection"] = f"kb_{self._kb_id}_documents"
                return MongoDBStorage(mongodb_config)
            else:
                # Use DatabaseFactory for other storage types
                return DatabaseFactory.create(storage_config)
        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            raise
    
    def _save_documents_to_storage(self):
        """
        Save documents to the storage backend.
        """
        if not self._storage:
            return
        
        try:
            # Convert documents to storage format
            storage_docs = []
            for doc in self._documents:
                storage_doc = {
                    "kb_id": self._kb_id,
                    "doc_id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "embeddings": doc.embeddings.tolist() if doc.embeddings is not None else None,
                    "topic_id": getattr(doc, 'topic_id', None),
                    "reduced_embeddings": getattr(doc, 'reduced_embeddings', None),
                                        "created_at": datetime.now(timezone.utc).isoformat(),
                    "language": self._language
                }
                storage_docs.append(storage_doc)
            
            # Store documents
            success = self._storage.store(storage_docs)
            if success:
                logger.info(f"Saved {len(storage_docs)} documents to storage")
            else:
                logger.error("Failed to save documents to storage")
                
        except Exception as e:
            logger.error(f"Error saving documents to storage: {e}")
    
    def _load_documents_from_storage(self) -> List[Document]:
        """
        Load documents from the storage backend.
        
        Returns:
            List of Document objects
        """
        if not self._storage:
            return []
        
        try:
            # Query documents for this knowledge base
            query = {"kb_id": self._kb_id}
            storage_docs = self._storage.query(query=query, limit=10000)  # Large limit for now
            
            documents = []
            for storage_doc in storage_docs:
                # Create Document object
                doc = Document(
                    content=storage_doc["content"],
                    metadata=storage_doc.get("metadata", {}),
                    id=storage_doc["doc_id"]
                )
                
                # Restore embeddings if available
                if storage_doc.get("embeddings"):
                    doc.embeddings = np.array(storage_doc["embeddings"])
                
                # Restore topic information
                if storage_doc.get("topic_id") is not None:
                    doc.topic_id = storage_doc["topic_id"]
                
                # Restore reduced embeddings
                if storage_doc.get("reduced_embeddings"):
                    doc.reduced_embeddings = storage_doc["reduced_embeddings"]
                
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from storage")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from storage: {e}")
            return []
    
    def _save_embeddings_to_storage(self):
        """
        Save embeddings to storage for existing documents.
        """
        if not self._storage:
            return
        
        try:
            # Update documents with embeddings
            for doc in self._documents:
                if doc.embeddings is not None:
                    update_data = {
                        "$set": {
                            "embeddings": doc.embeddings.tolist(),
                                                        "updated_at": datetime.now(timezone.utc).isoformat()
                        }
                    }
                    
                    # Update in storage
                    if hasattr(self._storage, 'update_one'):
                        self._storage.update_one(
                            {"kb_id": self._kb_id, "doc_id": doc.id},
                            update_data
                        )
            
            logger.info("Saved embeddings to storage")
            
        except Exception as e:
            logger.error(f"Error saving embeddings to storage: {e}")
    
    def _save_topics_to_storage(self):
        """
        Save topic information to storage.
        """
        if not self._storage:
            return
        
        try:
            # Save topic assignments to documents
            for doc in self._documents:
                if hasattr(doc, 'topic_id'):
                    update_data = {
                        "$set": {
                            "topic_id": doc.topic_id,
                            "reduced_embeddings": getattr(doc, 'reduced_embeddings', None),
                                                        "updated_at": datetime.now(timezone.utc).isoformat()
                        }
                    }
                    
                    # Update in storage
                    if hasattr(self._storage, 'update_one'):
                        self._storage.update_one(
                            {"kb_id": self._kb_id, "doc_id": doc.id},
                            update_data
                        )
            
            # Save topic metadata to a separate collection
            if hasattr(self._storage, 'insert_one') and self._topics_cache:
                topics_collection = f"kb_{self._kb_id}_topics"
                for topic_id, topic_name in self._topics_cache.items():
                    topic_doc = {
                        "kb_id": self._kb_id,
                        "topic_id": topic_id,
                        "topic_name": topic_name,
                        "document_count": len(self.get_documents_by_topic(topic_id)),
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }
                    
                    self._storage.insert_one(topic_doc, collection_name=topics_collection)
            
            logger.info("Saved topic information to storage")
            
        except Exception as e:
            logger.error(f"Error saving topics to storage: {e}")
    
    def _detect_language(self) -> str:
        """
        Detect the language of the documents.
        
        Returns:
            str: The detected language code (e.g., "en", "fr").
        """
        # Sample documents for language detection
        sample_size = min(DEFAULT_LANGUAGE_DETECTION_SAMPLE_SIZE, len(self._documents))
        sample_docs = self._rng.choice(self._documents, size=sample_size, replace=False)
        
        # Detect language for each document
        languages = []
        for doc in sample_docs:
            # Use only the first N characters for faster detection
            text = doc.content[:DEFAULT_LANGUAGE_DETECTION_MAX_TEXT_LENGTH]
            lang = detect_language(text)
            languages.append(lang)
        
        # Count language occurrences
        lang_counts = {}
        for lang in languages:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        # Return the most common language, or "en" if no language is detected
        if not lang_counts:
            return "en"
        
        return max(lang_counts.items(), key=lambda x: x[1])[0]
    
    @property
    def documents(self) -> List[Document]:
        """
        Get all documents in the knowledge base, as a list of Document objects.
        Usage: 
        kb = KnowledgeBase(data, client)
        for doc in kb.documents:
            print(doc.content)
        Returns:
            List[Document]: All documents in the knowledge base.
        """
        return self._documents
    
    def get_all_documents(self) -> List[Document]:
        """
        Get all documents in the knowledge base.
        
        This method provides an alternative to the documents property.
        
        Returns:
            List[Document]: All documents in the knowledge base.
        """
        return self._documents
    
    @property
    def language(self) -> str:
        """
        Get the language of the knowledge base.
        
        Returns:
            str: The language code (e.g., "en", "fr").
        """
        return self._language
    
    # TODO option to send embeddings as a list. If it does not exist, run this function.
    @property
    def embeddings(self) -> np.ndarray:
        """
        Get the embeddings of all documents in the knowledge base.
        
        Returns:
            np.ndarray: The embeddings of all documents.
        """
        if self._embeddings_cache is not None:
            return self._embeddings_cache
            
        logger.info("Computing embeddings for knowledge base")
        
        # Get embeddings in batches
        all_embeddings = []
        total_batches = (len(self._documents) + self._batch_size - 1) // self._batch_size
        
        for batch_idx, batch in enumerate(batch_iterator(self._documents, self._batch_size)):
            logger.info(f"Processing batch {batch_idx+1}/{total_batches}")
            
            batch_texts = [doc.content for doc in batch]
            batch_embeddings = self._client.get_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        # Store embeddings in documents
        for doc, emb in zip(self._documents, all_embeddings):
            doc.embeddings = np.array(emb)
        
        # Cache embeddings
        self._embeddings_cache = np.array(all_embeddings)
        
        # Save embeddings to storage if configured
        if self._storage:
            self._save_embeddings_to_storage()
        
        return self._embeddings_cache
    
    #TODO add parameters for this function
    @property
    def reduced_embeddings(self) -> np.ndarray:
        """
        Get the reduced embeddings of all documents in the knowledge base.
        
        Returns:
            np.ndarray: The reduced embeddings of all documents.
        """
        if self._reduced_embeddings_cache is None:
            logger.info("Computing reduced embeddings for knowledge base")
            
            try:
                import umap
                
                # Create UMAP reducer
                reducer = umap.UMAP(
                    n_neighbors=50,
                    min_dist=0.5,
                    n_components=2,
                    random_state=42,
                    n_jobs=1,
                )
                
                # Reduce embeddings
                reduced = reducer.fit_transform(self.embeddings)
                
                # Store reduced embeddings in documents
                for doc, emb in zip(self._documents, reduced):
                    doc.reduced_embeddings = emb.tolist()
                
                # Cache reduced embeddings
                self._reduced_embeddings_cache = reduced
            
            except Exception as e:
                logger.error(f"Failed to compute reduced embeddings: {str(e)}")
                # Return empty array as fallback
                self._reduced_embeddings_cache = np.zeros((len(self._documents), 2))
        
        return self._reduced_embeddings_cache
    
    @property
    def topics(self) -> Dict[int, str]:
        """
        Get the topics of the knowledge base.
        
        Returns:
            Dict[int, str]: A dictionary mapping topic IDs to topic names.
        """
        if self._topics_cache is None:
            logger.info("Finding topics in knowledge base")
            self._topics_cache = self._find_topics()
        
        return self._topics_cache
    #TODO is there another method for this topic 
    def _find_topics(self) -> Dict[int, str]:
        """
        Find topics in the knowledge base.
        
        Returns:
            Dict[int, str]: A dictionary mapping topic IDs to topic names.
        """
        try:
            from hdbscan import HDBSCAN
            
            # Create HDBSCAN clusterer
            clusterer = HDBSCAN(
                min_cluster_size=self._min_topic_size,
                min_samples=3,
                metric="euclidean",
                cluster_selection_epsilon=0.0,
            )
            
            # Cluster documents
            clustering = clusterer.fit(self.reduced_embeddings)
            
            # Assign topic IDs to documents
            for i, doc in enumerate(self._documents):
                doc.topic_id = int(clustering.labels_[i])
            
            # Get unique topic IDs
            topic_ids = set(clustering.labels_)
            
            # Generate topic names
            topics = {}
            for topic_id in topic_ids:
                if topic_id == -1:
                    # -1 is the noise cluster
                    topics[topic_id] = "Other"
                else:
                    # Get documents in this topic
                    topic_docs = [doc for doc in self._documents if doc.topic_id == topic_id]
                    # Generate topic name
                    topic_name = self._generate_topic_name(topic_docs)
                    topics[topic_id] = topic_name
            
            logger.info(f"Found {len(topics)} topics in knowledge base")
            
            # Save topics to storage if configured
            if self._storage:
                self._save_topics_to_storage()
            
            return topics
        
        except Exception as e:
            logger.error(f"Failed to find topics: {str(e)}")
            # Return a single "Unknown" topic as fallback
            for doc in self._documents:
                doc.topic_id = 0
            return {0: "Unknown"}
    
    def _generate_topic_name(self, topic_documents: List[Document]) -> str:
        """
        Generate a name for a topic.
        
        Args:
            topic_documents: The documents in the topic.
            
        Returns:
            str: The generated topic name.
        """
        # Shuffle documents to get a random sample
        self._rng.shuffle(topic_documents)
        
        # Get a sample of documents
        sample_size = min(10, len(topic_documents))
        sample_docs = topic_documents[:sample_size]
        
        # Create prompt
        topics_str = "\n\n".join(["----------" + doc.content[:500] for doc in sample_docs])
        
        # Prevent context window overflow
        topics_str = topics_str[:3 * 8192]
        
        prompt = TOPIC_GENERATION_PROMPT.safe_format(
            language=self._language,
            topics_elements=topics_str
        )
        
        try:
            # Generate topic name
            topic_name = self._client.generate_text(prompt)
            
            # Clean up topic name
            topic_name = topic_name.strip().strip('"')
            
            if not topic_name:
                logger.warning("Generated empty topic name, using fallback")
                return "Unknown Topic"
            
            logger.info(f"Generated topic name: {topic_name}")
            return topic_name
        
        except Exception as e:
            logger.error(f"Failed to generate topic name: {str(e)}")
            return "Unknown Topic"

    def add_documents(self, documents: List[Union[str, Document]], auto_embed: bool = True):
        """Adds documents to the knowledge base."""
        # Convert string documents to Document objects
        new_docs = [doc if isinstance(doc, Document) else Document(content=doc) for doc in documents]

        # Add to internal list
        self._documents.extend(new_docs)

        # Save to storage
        if self._storage:
            self._save_documents_to_storage(new_docs)

        # Trigger embedding if auto_embed is True
        if auto_embed:
            self.embed_documents()

    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            doc_id: The ID of the document to get.
            
        Returns:
            Optional[Document]: The document, or None if not found.
        """
        return self._document_index.get(doc_id)
    
    def get_documents_by_topic(self, topic_id: int) -> List[Document]:
        """
        Get all documents in a topic.
        
        Args:
            topic_id: The ID of the topic to get documents for.
            
        Returns:
            List[Document]: The documents in the topic.
        """
        return [doc for doc in self._documents if doc.topic_id == topic_id]
    
    def get_random_document(self) -> Document:
        """
        Get a random document from the knowledge base.
        
        Returns:
            Document: A random document.
        """
        return self._rng.choice(self._documents)
    
    def get_random_documents(self, n: int, with_replacement: bool = False) -> List[Document]:
        """
        Get random documents from the knowledge base.
        
        Args:
            n: The number of documents to get.
            with_replacement: Whether to allow the same document to be selected multiple times.
            
        Returns:
            List[Document]: The random documents.
        """
        if with_replacement or n > len(self._documents):
            return list(self._rng.choice(self._documents, n, replace=True))
        else:
            return list(self._rng.choice(self._documents, n, replace=False))
    def get_representative_documents(self, n: int, exclude_noise: bool = True) -> List[Document]:
        """
        Get a representative sample of documents based on topic distribution.
        
        This method samples documents from each topic proportionally to the
        number of documents in that topic, ensuring the sample represents
        the knowledge base structure.
        
        Args:
            n: The number of documents to sample.
            exclude_noise: Whether to exclude documents from the noise topic (topic_id=-1).
            
        Returns:
            List[Document]: The sampled documents.
        """
        # Use the new sample_documents_with_input_allocation method and extract just the documents
        doc_input_pairs = self.sample_documents_with_input_allocation(
            n=n,
            strategy="representative"
        )
        return [doc for doc, _ in doc_input_pairs]
    
    def sample_documents_with_input_allocation(
        self, 
        n: int, 
        strategy: str,
        items_per_document: int = 1
    ) -> List[Tuple[Document, int]]:
        """
        Sample documents from the knowledge base and determine input allocation.
        
        This method samples documents according to the specified strategy and
        determines how many test inputs should be generated for each document.
        
        Args:
            n: The total number of test inputs to generate.
            strategy: The sampling strategy to use:
                - "random": Randomly sample documents.
                - "representative": Sample documents proportionally to topic distribution.
                - "per_document": Sample documents and generate a fixed number of inputs per document.
            items_per_document: Number of items to generate per document.
                Only used when strategy is "per_document".
                
        Returns:
            List[Tuple[Document, int]]: List of (document, inputs_to_generate) tuples.
        """
        if strategy == "random":
            return self._sample_random_with_allocation(n)
        elif strategy == "representative":
            return self._sample_representative_with_allocation(n)
        elif strategy == "per_document":
            return self._sample_per_document_with_allocation(n, items_per_document)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def _sample_random_with_allocation(self, n: int) -> List[Tuple[Document, int]]:
        """
        Randomly sample documents and determine input allocation.
        
        Args:
            n: The total number of test inputs to generate.
            
        Returns:
            List[Tuple[Document, int]]: List of (document, inputs_to_generate) tuples.
        """
        # Determine how many documents to sample
        docs_to_sample = min(n, len(self._documents))
        
        # Sample documents
        sampled_docs = self.get_random_documents(docs_to_sample)
        
        # Calculate inputs per document
        base_inputs_per_doc = n // docs_to_sample
        extra_inputs = n % docs_to_sample
        
        # Allocate inputs to documents
        result = []
        for i, doc in enumerate(sampled_docs):
            inputs_for_doc = base_inputs_per_doc
            if i < extra_inputs:
                inputs_for_doc += 1
            result.append((doc, inputs_for_doc))
        
        return result

    def _sample_representative_with_allocation(self, n: int) -> List[Tuple[Document, int]]:
        """
        Sample documents proportionally to topic distribution and determine input allocation.
        
        Args:
            n: The total number of test inputs to generate.
            
        Returns:
            List[Tuple[Document, int]]: List of (document, inputs_to_generate) tuples.
        """
        # Get all topics
        topics = self.topics
        
        # Filter out noise topic
        valid_topics = {topic_id: name for topic_id, name in topics.items() if topic_id != -1}
        
        if not valid_topics:
            logger.warning("No valid topics found, falling back to random sampling")
            return self._sample_random_with_allocation(n)
        
        # Calculate the total number of documents in valid topics
        topic_document_counts = {}
        total_documents = 0
        
        for topic_id in valid_topics:
            topic_docs = self.get_documents_by_topic(topic_id)
            topic_document_counts[topic_id] = len(topic_docs)
            total_documents += len(topic_docs)
        
        # Calculate how many inputs to generate for each topic
        # based on the proportion of documents in that topic
        topic_input_counts = {}
        remaining_inputs = n
        
        # First, ensure each topic gets at least one input
        for topic_id in valid_topics:
            topic_input_counts[topic_id] = 1
            remaining_inputs -= 1
        
        # If we've already allocated all inputs, we're done
        if remaining_inputs <= 0:
            remaining_inputs = 0
        else:
            # Distribute remaining inputs proportionally
            for topic_id, doc_count in topic_document_counts.items():
                proportion = doc_count / total_documents
                additional_inputs = int(proportion * remaining_inputs)
                topic_input_counts[topic_id] += additional_inputs
                remaining_inputs -= additional_inputs
        
        # If we have any remaining inputs due to rounding,
        # distribute them to the largest topics
        if remaining_inputs > 0:
            sorted_topics = sorted(
                topic_document_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for topic_id, _ in sorted_topics:
                if remaining_inputs <= 0:
                    break
                topic_input_counts[topic_id] += 1
                remaining_inputs -= 1
        
        # Now determine how many documents to sample from each topic
        # and how many inputs to generate per document
        result = []
        
        for topic_id, input_count in topic_input_counts.items():
            topic_docs = self.get_documents_by_topic(topic_id)
            
            # Determine how many documents to sample from this topic
            # We want to maximize document diversity, so we'll use as many
            # documents as possible (up to the number of inputs needed)
            docs_to_sample = min(len(topic_docs), input_count)
            
            # Sample documents from this topic
            sampled_indices = self._rng.choice(
                len(topic_docs),
                size=docs_to_sample,
                replace=False
            )
            sampled_docs = [topic_docs[i] for i in sampled_indices]
            
            # Calculate inputs per document
            base_inputs_per_doc = input_count // docs_to_sample
            extra_inputs = input_count % docs_to_sample
            
            # Distribute inputs across documents
            for i, doc in enumerate(sampled_docs):
                # Give extra inputs to the first 'extra_inputs' documents
                inputs_for_this_doc = base_inputs_per_doc
                if i < extra_inputs:
                    inputs_for_this_doc += 1
                    
                result.append((doc, inputs_for_this_doc))
        
        # Shuffle the result to avoid clustering by topic
        self._rng.shuffle(result)
        
        return result

    def _sample_per_document_with_allocation(self, n: int, items_per_document: int) -> List[Tuple[Document, int]]:
        """
        Sample documents and allocate a fixed number of inputs per document.
        
        Args:
            n: The total number of test inputs to generate.
            items_per_document: Number of items to generate per document.
            
        Returns:
            List[Tuple[Document, int]]: List of (document, inputs_to_generate) tuples.
        """
        # Calculate how many documents to sample
        docs_to_sample = (n + items_per_document - 1) // items_per_document
        docs_to_sample = min(docs_to_sample, len(self._documents))
        
        # Sample documents
        sampled_docs = self.get_random_documents(docs_to_sample)
        
        # Allocate inputs to documents
        result = []
        remaining_inputs = n
        
        for doc in sampled_docs:
            # Allocate items to this document
            items_for_doc = min(items_per_document, remaining_inputs)
            if items_for_doc > 0:
                result.append((doc, items_for_doc))
                remaining_inputs -= items_for_doc
            
            # Stop if we've allocated all inputs
            if remaining_inputs <= 0:
                break
        
        return result
    def search(self, query: str, k: int = 5, mode: str = "embedding") -> List[Tuple[Document, float]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The query to search for.
            k: The number of results to return.
            mode: The search mode to use. Options:
                - "embedding": Use embeddings for semantic search (default)
                - "keyword": Use keyword matching (not yet implemented)
                - "hybrid": Use both embedding and keyword search (not yet implemented)
            
        Returns:
            List[Tuple[Document, float]]: The search results, as (document, score) pairs.
            
        Raises:
            ValueError: If an unsupported search mode is specified.
        """
        if mode == "embedding":
            return self._embedding_search(query, k)
        elif mode == "keyword":
            # TODO: Implement keyword search
            logger.warning("Keyword search not yet implemented, falling back to embedding search")
            return self._embedding_search(query, k)
        elif mode == "hybrid":
            # TODO: Implement hybrid search
            logger.warning("Hybrid search not yet implemented, falling back to embedding search")
            return self._embedding_search(query, k)
        else:
            raise ValueError(f"Unsupported search mode: {mode}")
    
    def _embedding_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to the query using embeddings.
        
        Args:
            query: The query to search for.
            k: The number of results to return.
            
        Returns:
            List[Tuple[Document, float]]: The search results, as (document, score) pairs.
        """
        # Get query embedding
        query_embedding = self._client.get_embeddings(query)[0]
        
        # Generate embeddings for documents if they don't exist
        if self._embeddings_cache is None:
            _ = self.embeddings  # This will generate and cache embeddings
        
        similarities = []
        for doc in self._documents:
            if doc.embeddings is None:
                # Skip documents without embeddings
                continue
            
            # Calculate cosine similarity
            similarity = cosine_similarity(np.array(query_embedding), np.array(doc.embeddings))
            similarities.append((doc, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return similarities[:k]
    
    def __len__(self) -> int:
        """
        Get the number of documents in the knowledge base.
        
        Returns:
            int: The number of documents.
        """
        return len(self._documents)
    
    def __getitem__(self, doc_id: str) -> Document:
        """
        Get a document by ID.
        
        Args:
            doc_id: The ID of the document to get.
            
        Returns:
            Document: The document.
            
        Raises:
            KeyError: If the document is not found.
        """
        doc = self.get_document(doc_id)
        if doc is None:
            raise KeyError(f"Document with ID {doc_id} not found")
        return doc
