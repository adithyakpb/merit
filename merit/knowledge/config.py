# /Users/adithyakp/Documents/Projects/merit/merit/knowledge/config.py
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
from pathlib import Path

class KnowledgeBaseBaseConfig(BaseModel):
    type: str

class KnowledgeBaseCsvConfig(KnowledgeBaseBaseConfig):
    type: Literal["csv"] = "csv"
    path: Path
    encoding: Optional[str] = "utf-8"
    delimiter: Optional[str] = ","

class KnowledgeBaseJsonConfig(KnowledgeBaseBaseConfig):
    type: Literal["json", "jsonl"] = "json"
    path: Path
    json_path_to_documents: Optional[str] = Field(None, description="JSONPath to the list of documents if nested, e.g., '$.docs[*]'")

class KnowledgeBaseVectorStoreConfig(KnowledgeBaseBaseConfig):
    type: Literal["vectorstore"] = "vectorstore"
    path: Optional[Path] = Field(None, description="Path to local vector store (e.g., ChromaDB, FAISS index file)")
    url: Optional[str] = Field(None, description="URL for remote vector store (e.g., Qdrant, Weaviate)")
    collection_name: Optional[str] = Field(None, description="Name of the collection or index in the vector store.")
    # Add other relevant params like client_type (e.g. qdrant, chroma)

KnowledgeBaseConfig = Union[KnowledgeBaseCsvConfig, KnowledgeBaseJsonConfig, KnowledgeBaseVectorStoreConfig]
