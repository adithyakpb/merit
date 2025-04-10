"""
Merit Knowledge Base

This module provides the knowledge base implementation for the Merit system.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple, Sequence

from ..api.base import BaseAPIClient
from ..core.models import Document
from .prompts import TOPIC_GENERATION_PROMPT
from ..core.utils import detect_language, cosine_similarity, batch_iterator
from ..core.logging import get_logger

logger = get_logger(__name__)

# Constants
DEFAULT_BATCH_SIZE = 32
DEFAULT_MIN_TOPIC_SIZE = 3
DEFAULT_LANGUAGE_DETECTION_SAMPLE_SIZE = 10
DEFAULT_LANGUAGE_DETECTION_MAX_TEXT_LENGTH = 300

#TODO enable connection to milvus KB, Mongo db, sqlite db
#TODO enable creation of knowledgebase from set of URLs
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
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        client: BaseAPIClient,
        columns: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        min_topic_size: Optional[int] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Initialize the knowledge base.
        
        Args:
            data: The data to create the knowledge base from, either a pandas DataFrame or a list of dictionaries.
            client: The API client to use for embeddings and text generation.
            columns: The columns to use from the data. If None, all columns are used.
            seed: The random seed to use.
            min_topic_size: The minimum number of documents to form a topic.
            batch_size: The batch size to use for embeddings.
        """
        # Convert data to DataFrame if it's a list
        if isinstance(data, list):
            data = pd.DataFrame(data)
        
        if len(data) == 0:
            raise ValueError("Cannot create a knowledge base from empty data")
        
        # Set up random number generator
        self._rng = np.random.default_rng(seed=seed)
        
        # Store parameters
        self._client = client
        self._batch_size = batch_size
        self._min_topic_size = min_topic_size or DEFAULT_MIN_TOPIC_SIZE
        
        # Create documents
        self._documents = self._create_documents(data, columns)
        
        if len(self._documents) == 0:
            raise ValueError("Cannot create a knowledge base with empty documents")
        
        # Create document index
        self._document_index = {doc.id: doc for doc in self._documents}
        
        # Initialize caches
        self._embeddings_cache = None
        self._topics_cache = None
        self._index_cache = None
        self._reduced_embeddings_cache = None
        
        # Detect language
        self._language = self._detect_language()
        
        logger.info(f"Created knowledge base with {len(self._documents)} documents in language '{self._language}'")
    
    def _create_documents(self, data: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> List[Document]:
        """
        Create documents from the data.
        
        Args:
            data: The data to create documents from.
            columns: The columns to use from the data. If None, all columns are used.
            
        Returns:
            List[Document]: The created documents.
        """
        # If columns is None, use all columns
        if columns is None:
            columns = data.columns.tolist()
        
        # Create documents
        documents = []
        for idx, row in data.iterrows():
            # Create content by joining columns
            if len(columns) > 1:
                content = "\n".join(f"{col}: {row[col]}" for col in columns if col in row)
            else:
                content = str(row[columns[0]])
            
            # Skip empty documents
            if not content.strip():
                continue
            
            # Create document
            doc = Document(
                content=content,
                metadata=row.to_dict(),
                id=str(idx),
            )
            documents.append(doc)
        
        return documents
    
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
            doc.embeddings = emb
        
        # Cache embeddings
        self._embeddings_cache = np.array(all_embeddings)
        
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
