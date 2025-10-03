import os
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import json
from .config import DB_CONFIG, SEARCH_CONFIG

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Advanced vector store with multiple collections and optimizations"""
    
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or DB_CONFIG.VECTOR_DB_PATH
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded embedding model: all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Track collections
        self.collections = {}
        self._initialize_collections()
    
    def _initialize_collections(self):
        """Initialize or load existing collections"""
        collection_names = DB_CONFIG.COLLECTION_NAMES or [
            "technical_docs", "business_knowledge", "research_papers", "general_knowledge"
        ]
        
        for name in collection_names:
            try:
                # Try to get existing collection
                collection = self.client.get_collection(name=name)
                logger.info(f"Loaded existing collection: {name}")
            except ValueError:
                # Create new collection if it doesn't exist
                collection = self.client.create_collection(
                    name=name,
                    metadata={"description": f"Collection for {name.replace('_', ' ')}"}
                )
                logger.info(f"Created new collection: {name}")
            
            self.collections[name] = collection
    
    def add_documents(self, documents: List[Document], collection_name: str) -> int:
        """Add documents to specified collection with embeddings"""
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return 0
        
        if not documents:
            logger.warning("No documents to add")
            return 0
        
        collection = self.collections[collection_name]
        
        try:
            # Prepare data for ChromaDB
            doc_texts = []
            doc_metadatas = []
            doc_ids = []
            embeddings = []
            
            for i, doc in enumerate(documents):
                # Generate unique ID
                doc_id = self._generate_doc_id(doc, collection_name, i)
                
                # Skip if document already exists
                if self._document_exists(collection, doc_id):
                    logger.debug(f"Document {doc_id} already exists, skipping")
                    continue
                
                # Prepare text and metadata
                doc_texts.append(doc.page_content)
                
                # Clean metadata for ChromaDB (must be JSON serializable)
                clean_metadata = self._clean_metadata(doc.metadata)
                doc_metadatas.append(clean_metadata)
                
                doc_ids.append(doc_id)
            
            if not doc_texts:
                logger.info("All documents already exist in collection")
                return 0
            
            # Generate embeddings in batches for efficiency
            logger.info(f"Generating embeddings for {len(doc_texts)} documents")
            embeddings = self.embedding_model.encode(
                doc_texts, 
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
            
            # Add to ChromaDB
            collection.add(
                embeddings=embeddings,
                documents=doc_texts,
                metadatas=doc_metadatas,
                ids=doc_ids
            )
            
            logger.info(f"Added {len(doc_texts)} documents to collection {collection_name}")
            return len(doc_texts)
            
        except Exception as e:
            logger.error(f"Error adding documents to {collection_name}: {e}")
            return 0
    
    def similarity_search(
        self, 
        query: str, 
        collection_name: str, 
        top_k: int = None,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search in specified collection"""
        top_k = top_k or SEARCH_CONFIG.DEFAULT_TOP_K
        
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return []
        
        collection = self.collections[collection_name]
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Perform search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else None,
                        'id': results['ids'][0][i],
                        'collection': collection_name
                    }
                    formatted_results.append(result)
            
            logger.debug(f"Found {len(formatted_results)} results in {collection_name}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in similarity search for {collection_name}: {e}")
            return []
    
    def multi_collection_search(
        self, 
        query: str, 
        collections: List[str] = None, 
        top_k_per_collection: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search across multiple collections"""
        collections = collections or list(self.collections.keys())
        
        results = {}
        for collection_name in collections:
            if collection_name in self.collections:
                search_results = self.similarity_search(
                    query=query,
                    collection_name=collection_name,
                    top_k=top_k_per_collection
                )
                results[collection_name] = search_results
            else:
                logger.warning(f"Collection {collection_name} not found")
                results[collection_name] = []
        
        return results
    
    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all collections"""
        stats = {}
        
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                # Get a sample of documents to analyze
                sample_results = collection.query(
                    query_embeddings=[[0.0] * 384],  # Zero vector for random sampling
                    n_results=min(10, count) if count > 0 else 0
                )
                
                stats[name] = {
                    'document_count': count,
                    'sample_metadata': sample_results.get('metadatas', [])[:3] if sample_results else []
                }
            except Exception as e:
                logger.warning(f"Error getting stats for {name}: {e}")
                stats[name] = {'document_count': 0, 'sample_metadata': []}
        
        return stats
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            if collection_name in self.collections:
                self.client.delete_collection(collection_name)
                del self.collections[collection_name]
                logger.info(f"Deleted collection: {collection_name}")
                return True
            else:
                logger.warning(f"Collection {collection_name} not found")
                return False
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False
    
    def reset_all_collections(self) -> bool:
        """Reset all collections (use with caution)"""
        try:
            self.client.reset()
            self.collections = {}
            self._initialize_collections()
            logger.info("Reset all collections")
            return True
        except Exception as e:
            logger.error(f"Error resetting collections: {e}")
            return False
    
    def _generate_doc_id(self, doc: Document, collection_name: str, index: int) -> str:
        """Generate unique document ID"""
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:16]
        source = doc.metadata.get('source', 'unknown')
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        return f"{collection_name}_{source_hash}_{content_hash}_{index}"
    
    def _document_exists(self, collection, doc_id: str) -> bool:
        """Check if document already exists in collection"""
        try:
            results = collection.get(ids=[doc_id])
            return len(results['ids']) > 0
        except Exception:
            return False
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata for ChromaDB storage"""
        clean_meta = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean_meta[key] = value
            elif isinstance(value, list):
                # Convert list to JSON string
                clean_meta[key] = json.dumps(value)
            elif isinstance(value, dict):
                # Convert dict to JSON string  
                clean_meta[key] = json.dumps(value)
            else:
                # Convert other types to string
                clean_meta[key] = str(value)
        
        return clean_meta
    
    def hybrid_search(
        self, 
        query: str, 
        collection_name: str, 
        top_k: int = None,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and keyword matching"""
        top_k = top_k or SEARCH_CONFIG.DEFAULT_TOP_K
        
        # Get semantic search results
        semantic_results = self.similarity_search(query, collection_name, top_k * 2)
        
        # Simple keyword scoring
        query_words = set(query.lower().split())
        
        for result in semantic_results:
            doc_words = set(result['document'].lower().split())
            keyword_score = len(query_words & doc_words) / len(query_words)
            
            # Combine scores (assuming distance is similarity-based)
            semantic_score = 1 - (result.get('distance', 0.5))
            combined_score = (1 - keyword_weight) * semantic_score + keyword_weight * keyword_score
            
            result['combined_score'] = combined_score
        
        # Sort by combined score and return top_k
        hybrid_results = sorted(semantic_results, key=lambda x: x['combined_score'], reverse=True)
        return hybrid_results[:top_k]