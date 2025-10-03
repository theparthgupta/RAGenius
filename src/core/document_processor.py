"""
Document processing engine for the Advanced RAG System
Handles PDF, web content, and text file processing
"""
import os
import hashlib
import logging
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import spacy
from .config import SEARCH_CONFIG

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Advanced document processing with multiple input types"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=SEARCH_CONFIG.CHUNK_SIZE,
            chunk_overlap=SEARCH_CONFIG.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Load spacy model for advanced text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """Extract text from PDF and create document chunks"""
        try:
            documents = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                full_text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
                
                if full_text.strip():
                    # Create metadata
                    metadata = {
                        "source": file_path,
                        "type": "pdf",
                        "pages": len(pdf_reader.pages),
                        "file_hash": self._get_file_hash(file_path)
                    }
                    
                    # Split into chunks
                    chunks = self.text_splitter.split_text(full_text)
                    
                    for i, chunk in enumerate(chunks):
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_id"] = i
                        documents.append(Document(page_content=chunk, metadata=chunk_metadata))
                
                logger.info(f"Processed PDF: {file_path} -> {len(documents)} chunks")
                return documents
                
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return []
    
    def process_web_content(self, urls: List[str]) -> List[Document]:
        """Scrape and process web content from multiple URLs"""
        documents = []
        
        for url in urls:
            try:
                # Fetch content
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Parse content
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    element.decompose()
                
                # Extract clean text
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks_text = (phrase.strip() for line in lines for phrase in line.split("  "))
                clean_text = ' '.join(chunk for chunk in chunks_text if chunk)
                
                if clean_text and len(clean_text) > 100:  # Minimum content threshold
                    metadata = {
                        "source": url,
                        "type": "web",
                        "title": soup.title.string if soup.title else url,
                        "content_hash": hashlib.md5(clean_text.encode()).hexdigest()
                    }
                    
                    # Split into chunks
                    chunks = self.text_splitter.split_text(clean_text)
                    
                    for i, chunk in enumerate(chunks):
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_id"] = i
                        documents.append(Document(page_content=chunk, metadata=chunk_metadata))
                    
                    logger.info(f"Processed URL: {url} -> {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")
        
        return documents
    
    def process_text_file(self, file_path: str) -> List[Document]:
        """Process plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            if content.strip():
                metadata = {
                    "source": file_path,
                    "type": "text",
                    "file_hash": self._get_file_hash(file_path)
                }
                
                chunks = self.text_splitter.split_text(content)
                documents = []
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_id"] = i
                    documents.append(Document(page_content=chunk, metadata=chunk_metadata))
                
                logger.info(f"Processed text file: {file_path} -> {len(documents)} chunks")
                return documents
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
        
        return []
    
    def enhance_text_with_nlp(self, text: str) -> Dict[str, Any]:
        """Extract additional information using NLP"""
        if not self.nlp:
            return {"entities": [], "key_phrases": []}
        
        try:
            doc = self.nlp(text)
            
            # Extract named entities
            entities = [
                {"text": ent.text, "label": ent.label_, "description": spacy.explain(ent.label_)}
                for ent in doc.ents
            ]
            
            # Extract key noun phrases
            key_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
            
            return {
                "entities": entities[:10],  # Limit to top 10
                "key_phrases": key_phrases[:15]  # Limit to top 15
            }
            
        except Exception as e:
            logger.warning(f"NLP processing error: {e}")
            return {"entities": [], "key_phrases": []}
    
    def categorize_content(self, text: str) -> str:
        """Simple content categorization"""
        text_lower = text.lower()
        
        # Technical indicators
        tech_keywords = ['api', 'function', 'class', 'method', 'code', 'programming', 
                        'database', 'algorithm', 'framework', 'library']
        
        # Business indicators  
        business_keywords = ['policy', 'procedure', 'guideline', 'process', 'workflow',
                           'meeting', 'decision', 'strategy', 'management']
        
        # Research indicators
        research_keywords = ['study', 'research', 'analysis', 'experiment', 'hypothesis',
                           'conclusion', 'methodology', 'findings', 'literature']
        
        tech_score = sum(1 for kw in tech_keywords if kw in text_lower)
        business_score = sum(1 for kw in business_keywords if kw in text_lower)
        research_score = sum(1 for kw in research_keywords if kw in text_lower)
        
        if tech_score >= business_score and tech_score >= research_score:
            return "technical_docs"
        elif business_score >= research_score:
            return "business_knowledge"
        elif research_score > 0:
            return "research_papers"
        else:
            return "general_knowledge"
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file content"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return hashlib.md5(file_path.encode()).hexdigest()
    
    def process_batch(self, sources: List[Dict[str, Any]]) -> Dict[str, List[Document]]:
        """Process multiple sources and categorize them"""
        results = {
            "technical_docs": [],
            "business_knowledge": [],
            "research_papers": [],
            "general_knowledge": []
        }
        
        for source in sources:
            source_type = source.get("type")
            source_path = source.get("path")
            
            documents = []
            
            if source_type == "pdf":
                documents = self.process_pdf(source_path)
            elif source_type == "web":
                documents = self.process_web_content([source_path])
            elif source_type == "text":
                documents = self.process_text_file(source_path)
            
            # Categorize and distribute documents
            for doc in documents:
                category = self.categorize_content(doc.page_content)
                
                # Add NLP enhancements
                nlp_info = self.enhance_text_with_nlp(doc.page_content)
                doc.metadata.update(nlp_info)
                doc.metadata["category"] = category
                
                results[category].append(doc)
        
        # Log summary
        total_docs = sum(len(docs) for docs in results.values())
        logger.info(f"Batch processing complete: {total_docs} total documents")
        for category, docs in results.items():
            logger.info(f"  {category}: {len(docs)} documents")
        
        return results