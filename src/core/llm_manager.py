"""
Local LLM management for the Advanced RAG System
Handles response generation, prompt engineering, and model operations
"""
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import re
import json
import ollama
from .config import MODEL_CONFIG, SEARCH_CONFIG

logger = logging.getLogger(__name__)

class LocalLLMManager:
    """Advanced LLM management with local models via Ollama"""
    
    def __init__(self):
        self.primary_model = MODEL_CONFIG.LLM_MODEL
        self.backup_model = MODEL_CONFIG.BACKUP_LLM_MODEL
        self.current_model = self.primary_model
        
        # Test model availability
        self._check_model_availability()
        
        # Response templates and prompts
        self.prompt_templates = self._initialize_prompt_templates()
        
        # Generation parameters
        self.generation_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "stop": ["Human:", "Assistant:", "User:", "AI:"]
        }
    
    def _check_model_availability(self):
        """Check if required models are available"""
        try:
            # List available models
            available_models = ollama.list()
            model_names = [model['name'] for model in available_models['models']]
            
            if self.primary_model in model_names:
                logger.info(f"Primary model available: {self.primary_model}")
            elif self.backup_model in model_names:
                logger.warning(f"Primary model not found, using backup: {self.backup_model}")
                self.current_model = self.backup_model
            else:
                logger.error(f"Neither primary ({self.primary_model}) nor backup ({self.backup_model}) models available")
                raise Exception("No suitable models available")
                
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            raise
    
    def _initialize_prompt_templates(self) -> Dict[str, str]:
        """Initialize prompt templates for different scenarios"""
        return {
            "standard_rag": """Based on the following context information, please answer the user's question accurately and concisely.

Context:
{context}

Question: {question}

Instructions:
- Use only the information provided in the context
- If the context doesn't contain enough information to answer the question, say so clearly
- Cite specific sources when possible
- Be concise but comprehensive
- Maintain a helpful and professional tone

Answer:""",

            "comparison_query": """Based on the following context from multiple sources, please provide a comprehensive comparison to answer the user's question.

Context Sources:
{context}

Question: {question}

Instructions:
- Compare information from different sources
- Highlight similarities and differences
- Cite specific sources for each point
- If sources contradict, mention the discrepancy
- Provide a balanced perspective

Answer:""",

            "technical_query": """As a technical expert, please answer the following question using the provided context.

Technical Context:
{context}

Question: {question}

Instructions:
- Focus on technical accuracy
- Include code examples if relevant
- Explain technical concepts clearly
- Mention any prerequisites or dependencies
- Provide step-by-step guidance when appropriate

Answer:""",

            "research_query": """Based on the academic/research context provided, please answer the question with appropriate academic rigor.

Research Context:
{context}

Question: {question}

Instructions:
- Maintain academic tone
- Reference specific studies or findings
- Discuss methodology when relevant
- Mention limitations or uncertainties
- Use appropriate academic terminology

Answer:""",

            "business_query": """Using the business context provided, please answer the question from a business perspective.

Business Context:
{context}

Question: {question}

Instructions:
- Focus on practical business implications
- Consider stakeholder perspectives
- Mention relevant policies or procedures
- Provide actionable insights
- Use clear, professional language

Answer:"""
        }
    
    def generate_response(
        self, 
        query: str, 
        context_docs: List[Dict[str, Any]], 
        query_type: str = "general",
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """Generate response using local LLM with context"""
        
        start_time = time.time()
        
        try:
            # Prepare context
            context = self._prepare_context(context_docs, query_type)
            
            # Select appropriate prompt template
            prompt_template = self._select_prompt_template(query_type, context_docs)
            
            # Build final prompt
            prompt = prompt_template.format(
                context=context,
                question=query
            )
            
            # Generate response
            response = self._call_ollama(prompt, max_tokens)
            
            # Extract citations
            citations = self._extract_citations(response, context_docs)
            
            # Calculate metrics
            generation_time = time.time() - start_time
            
            result = {
                "response": response,
                "citations": citations,
                "context_used": len(context_docs),
                "generation_time": generation_time,
                "model_used": self.current_model,
                "query_type": query_type,
                "prompt_length": len(prompt),
                "response_length": len(response)
            }
            
            logger.debug(f"Generated response in {generation_time:.2f}s using {self.current_model}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": f"Sorry, I encountered an error while generating a response: {str(e)}",
                "citations": [],
                "context_used": 0,
                "generation_time": time.time() - start_time,
                "model_used": self.current_model,
                "error": str(e)
            }
    
    def _prepare_context(self, context_docs: List[Dict[str, Any]], query_type: str) -> str:
        """Prepare context string from retrieved documents"""
        if not context_docs:
            return "No relevant context found."
        
        context_parts = []
        
        for i, doc in enumerate(context_docs[:5]):  # Limit to top 5 documents
            # Extract document info
            content = doc.get('document', '')
            metadata = doc.get('metadata', {})
            collection = doc.get('collection', 'unknown')
            
            # Get source information
            source = metadata.get('source', 'Unknown source')
            chunk_id = metadata.get('chunk_id', i)
            
            # Format context entry
            context_entry = f"""
Source {i+1} ({collection}): {source}
Content: {content[:800]}{'...' if len(content) > 800 else ''}
---"""
            
            context_parts.append(context_entry)
        
        return "\n".join(context_parts)
    
    def _select_prompt_template(self, query_type: str, context_docs: List[Dict[str, Any]]) -> str:
        """Select appropriate prompt template based on query type and context"""
        
        # Determine if it's a comparison query
        if len(set(doc.get('collection', '') for doc in context_docs)) > 1:
            return self.prompt_templates["comparison_query"]
        
        # Select based on collection type
        collections = [doc.get('collection', '') for doc in context_docs]
        primary_collection = max(set(collections), key=collections.count) if collections else ""
        
        if primary_collection == "technical_docs":
            return self.prompt_templates["technical_query"]
        elif primary_collection == "research_papers":
            return self.prompt_templates["research_query"]
        elif primary_collection == "business_knowledge":
            return self.prompt_templates["business_query"]
        else:
            return self.prompt_templates["standard_rag"]
    
    def _call_ollama(self, prompt: str, max_tokens: int) -> str:
        """Make API call to Ollama"""
        try:
            response = ollama.generate(
                model=self.current_model,
                prompt=prompt,
                options={
                    **self.generation_params,
                    "num_predict": max_tokens
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            
            # Try backup model if primary fails
            if self.current_model == self.primary_model:
                try:
                    logger.info("Trying backup model...")
                    response = ollama.generate(
                        model=self.backup_model,
                        prompt=prompt,
                        options={
                            **self.generation_params,
                            "num_predict": max_tokens
                        }
                    )
                    self.current_model = self.backup_model
                    return response['response'].strip()
                except Exception as backup_error:
                    logger.error(f"Backup model also failed: {backup_error}")
            
            raise e
    
    def _extract_citations(self, response: str, context_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and format citations from context documents"""
        citations = []
        
        for i, doc in enumerate(context_docs):
            metadata = doc.get('metadata', {})
            
            citation = {
                "id": i + 1,
                "source": metadata.get('source', 'Unknown'),
                "type": metadata.get('type', 'document'),
                "collection": doc.get('collection', 'unknown'),
                "relevance_score": 1 - doc.get('distance', 0.5) if doc.get('distance') else 0.5
            }
            
            # Add additional metadata based on document type
            if metadata.get('type') == 'pdf':
                citation['pages'] = metadata.get('pages')
            elif metadata.get('type') == 'web':
                citation['title'] = metadata.get('title')
            
            citations.append(citation)
        
        return citations
    
    def summarize_documents(self, documents: List[str], max_length: int = 200) -> str:
        """Generate a summary of multiple documents"""
        
        if not documents:
            return "No documents to summarize."
        
        # Combine documents
        combined_text = "\n\n".join(documents[:5])  # Limit to 5 documents
        
        prompt = f"""Please provide a concise summary of the following text:

Text to summarize:
{combined_text[:3000]}  

Instructions:
- Create a summary of no more than {max_length} words
- Focus on the main points and key information
- Use clear and concise language
- Maintain the original meaning

Summary:"""
        
        try:
            response = ollama.generate(
                model=self.current_model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": max_length + 50
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Unable to generate summary at this time."
    
    def evaluate_response_quality(self, response: str, query: str, context: str) -> Dict[str, float]:
        """Simple heuristic evaluation of response quality"""
        
        scores = {}
        
        # Length appropriateness (not too short, not too long)
        response_length = len(response.split())
        if 10 <= response_length <= 300:
            scores['length_score'] = 1.0
        elif response_length < 10:
            scores['length_score'] = response_length / 10.0
        else:
            scores['length_score'] = max(0.5, 1.0 - (response_length - 300) / 300)
        
        # Relevance (simple keyword matching)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if query_words:
            relevance = len(query_words & response_words) / len(query_words)
            scores['relevance_score'] = min(1.0, relevance * 2)  # Boost the score
        else:
            scores['relevance_score'] = 0.5
        
        # Context usage (check if response references context)
        context_indicators = [
            "according to", "based on", "the document", "the source", 
            "as mentioned", "as stated", "from the context"
        ]
        
        context_usage = sum(1 for indicator in context_indicators 
                          if indicator in response.lower())
        scores['context_usage_score'] = min(1.0, context_usage / 2)
        
        # Overall quality score
        scores['overall_score'] = (
            scores['length_score'] * 0.2 + 
            scores['relevance_score'] * 0.5 + 
            scores['context_usage_score'] * 0.3
        )
        
        return scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        try:
            models = ollama.list()
            return {
                "current_model": self.current_model,
                "primary_model": self.primary_model,
                "backup_model": self.backup_model,
                "available_models": [model['name'] for model in models['models']],
                "generation_params": self.generation_params
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                "current_model": self.current_model,
                "error": str(e)
            }
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        try:
            # Test the model with a simple prompt
            test_response = ollama.generate(
                model=model_name,
                prompt="Hello, this is a test.",
                options={"num_predict": 10}
            )
            
            self.current_model = model_name
            logger.info(f"Switched to model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch to model {model_name}: {e}")
            return False