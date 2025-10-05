"""
Intelligent query routing system for the Advanced RAG System
Routes queries to appropriate knowledge bases using classification
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import json
from .config import DB_CONFIG

logger = logging.getLogger(__name__)

class QueryRouter:
    """Intelligent query routing with multiple classification strategies"""
    
    def __init__(self):
        self.collection_names = DB_CONFIG.COLLECTION_NAMES or [
            "technical_docs", "business_knowledge", "research_papers", "general_knowledge"
        ]
        
        # Classification pipeline
        self.classifier = None
        self.is_trained = False
        
        # Rule-based routing patterns
        self.routing_patterns = self._initialize_routing_patterns()
        
        # Initialize ML classifier with default training data
        self._initialize_ml_classifier()
    
    def _initialize_routing_patterns(self) -> Dict[str, List[str]]:
        """Initialize rule-based routing patterns"""
        return {
            "technical_docs": [
                # Programming and development
                r"\b(?:api|function|method|class|code|programming|developer?|software)\b",
                r"\b(?:database|sql|query|table|schema|migration)\b",
                r"\b(?:framework|library|package|module|import)\b",
                r"\b(?:bug|error|exception|debug|troubleshoot|fix)\b",
                r"\b(?:algorithm|data structure|complexity|optimization)\b",
                r"\b(?:version|update|patch|release|deployment)\b",
                r"\b(?:authentication|authorization|security|encryption)\b",
                # Technical concepts
                r"\b(?:rest|http|json|xml|yaml|config|configuration)\b",
                r"\b(?:git|repository|commit|branch|merge|pull request)\b",
                r"\b(?:docker|container|kubernetes|microservice)\b"
            ],
            "business_knowledge": [
                # Business processes
                r"\b(?:policy|procedure|guideline|process|workflow)\b",
                r"\b(?:meeting|decision|strategy|planning|management)\b",
                r"\b(?:budget|cost|expense|revenue|profit|finance)\b",
                r"\b(?:customer|client|user|stakeholder|team)\b",
                r"\b(?:project|milestone|deadline|deliverable|timeline)\b",
                r"\b(?:requirement|specification|scope|objective)\b",
                r"\b(?:contract|agreement|terms|conditions|legal)\b",
                r"\b(?:hr|human resources|employee|staff|personnel)\b",
                r"\b(?:marketing|sales|promotion|campaign|brand)\b",
                r"\b(?:compliance|regulation|standard|audit|governance)\b"
            ],
            "research_papers": [
                # Academic and research terms
                r"\b(?:study|research|analysis|experiment|hypothesis)\b",
                r"\b(?:methodology|method|approach|technique|framework)\b",
                r"\b(?:conclusion|finding|result|outcome|discovery)\b",
                r"\b(?:literature|review|survey|meta-analysis)\b",
                r"\b(?:theory|model|paradigm|concept|principle)\b",
                r"\b(?:data|dataset|sample|population|statistics)\b",
                r"\b(?:publication|paper|article|journal|conference)\b",
                r"\b(?:abstract|introduction|discussion|references)\b",
                r"\b(?:significant|correlation|regression|p-value)\b",
                r"\b(?:academic|scholarly|peer.reviewed|citation)\b"
            ],
            "general_knowledge": [
                # General information patterns
                r"\b(?:what|who|when|where|why|how|explain|define)\b",
                r"\b(?:history|historical|background|origin|evolution)\b",
                r"\b(?:example|instance|case|scenario|situation)\b",
                r"\b(?:overview|summary|introduction|basics|fundamentals)\b",
                r"\b(?:news|current|recent|latest|update|trend)\b",
                r"\b(?:compare|comparison|difference|similarity|versus)\b",
                r"\b(?:benefit|advantage|disadvantage|pro|con)\b"
            ]
        }
    
    def _initialize_ml_classifier(self):
        """Initialize ML classifier with synthetic training data"""
        try:
            # Create synthetic training data
            training_data = self._generate_training_data()
            
            if training_data:
                # Create and train classifier pipeline
                self.classifier = Pipeline([
                    ('tfidf', TfidfVectorizer(
                        max_features=1000,
                        ngram_range=(1, 2),
                        stop_words='english',
                        lowercase=True
                    )),
                    ('nb', MultinomialNB(alpha=0.1))
                ])
                
                texts, labels = zip(*training_data)
                self.classifier.fit(texts, labels)
                self.is_trained = True
                
                logger.info(f"ML classifier trained with {len(training_data)} examples")
            
        except Exception as e:
            logger.warning(f"Failed to initialize ML classifier: {e}")
            self.classifier = None
            self.is_trained = False
    
    def _generate_training_data(self) -> List[Tuple[str, str]]:
        """Generate synthetic training data for classifier"""
        training_examples = [
            # Technical documentation examples
            ("How do I implement authentication in the API?", "technical_docs"),
            ("What's the database schema for users table?", "technical_docs"),
            ("Can you show me the code for data validation?", "technical_docs"),
            ("How to fix this SQL query error?", "technical_docs"),
            ("What frameworks are we using for frontend?", "technical_docs"),
            ("How do I deploy the application to production?", "technical_docs"),
            ("What's the API endpoint for user management?", "technical_docs"),
            ("How to handle exceptions in the code?", "technical_docs"),
            
            # Business knowledge examples
            ("What's our company policy on remote work?", "business_knowledge"),
            ("How do we handle customer complaints?", "business_knowledge"),
            ("What are the project deadlines for Q4?", "business_knowledge"),
            ("Can you explain our budget approval process?", "business_knowledge"),
            ("What's the procedure for onboarding new employees?", "business_knowledge"),
            ("How do we track project milestones?", "business_knowledge"),
            ("What are our compliance requirements?", "business_knowledge"),
            ("Who approves marketing campaign budgets?", "business_knowledge"),
            
            # Research papers examples
            ("What does the latest study say about machine learning?", "research_papers"),
            ("Can you summarize the research methodology used?", "research_papers"),
            ("What were the findings of the experiment?", "research_papers"),
            ("How was the data collected for this analysis?", "research_papers"),
            ("What's the statistical significance of the results?", "research_papers"),
            ("Can you explain the theoretical framework?", "research_papers"),
            ("What are the limitations mentioned in the study?", "research_papers"),
            ("How does this research compare to previous work?", "research_papers"),
            
            # General knowledge examples
            ("What is artificial intelligence?", "general_knowledge"),
            ("Can you explain how blockchain works?", "general_knowledge"),
            ("What's the difference between AI and ML?", "general_knowledge"),
            ("Give me an overview of cloud computing", "general_knowledge"),
            ("What are the benefits of remote work?", "general_knowledge"),
            ("How has technology evolved over time?", "general_knowledge"),
            ("What are some examples of automation?", "general_knowledge"),
            ("Explain the basics of data science", "general_knowledge")
        ]
        
        return training_examples
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """Route query to appropriate collection(s) with confidence scores"""
        
        # Clean and preprocess query
        clean_query = self._preprocess_query(query)
        
        # Get routing decisions from different methods
        rule_based_result = self._rule_based_routing(clean_query)
        ml_result = self._ml_based_routing(clean_query) if self.is_trained else None
        
        # Combine results
        routing_decision = self._combine_routing_decisions(
            rule_based_result, 
            ml_result,
            clean_query
        )
        
        return routing_decision
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better classification"""
        # Convert to lowercase
        query = query.lower().strip()
        
        # Remove special characters but keep spaces
        query = re.sub(r'[^\w\s]', ' ', query)
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)
        
        return query
    
    def _rule_based_routing(self, query: str) -> Dict[str, float]:
        """Rule-based routing using pattern matching"""
        scores = {collection: 0.0 for collection in self.collection_names}
        
        for collection, patterns in self.routing_patterns.items():
            if collection in scores:
                for pattern in patterns:
                    matches = len(re.findall(pattern, query, re.IGNORECASE))
                    scores[collection] += matches
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
        
        return scores
    
    def _ml_based_routing(self, query: str) -> Dict[str, float]:
        """ML-based routing using trained classifier"""
        if not self.classifier or not self.is_trained:
            return {collection: 0.25 for collection in self.collection_names}
        
        try:
            # Get prediction probabilities
            probabilities = self.classifier.predict_proba([query])[0]
            classes = self.classifier.classes_
            
            # Create score dictionary
            scores = {}
            for collection in self.collection_names:
                if collection in classes:
                    idx = list(classes).index(collection)
                    scores[collection] = probabilities[idx]
                else:
                    scores[collection] = 0.0
            
            return scores
            
        except Exception as e:
            logger.warning(f"ML routing failed: {e}")
            return {collection: 0.25 for collection in self.collection_names}
    
    def _combine_routing_decisions(
        self, 
        rule_scores: Dict[str, float], 
        ml_scores: Optional[Dict[str, float]], 
        query: str
    ) -> Dict[str, Any]:
        """Combine routing decisions from multiple methods"""
        
        # Weights for combining methods
        rule_weight = 0.6
        ml_weight = 0.4
        
        combined_scores = {}
        
        for collection in self.collection_names:
            rule_score = rule_scores.get(collection, 0.0)
            ml_score = ml_scores.get(collection, 0.25) if ml_scores else 0.25
            
            combined_scores[collection] = (
                rule_weight * rule_score + ml_weight * ml_score
            )
        
        # Find the best collection(s)
        max_score = max(combined_scores.values())
        
        # If no clear winner, route to multiple collections
        threshold = 0.7  # Confidence threshold
        
        if max_score < threshold:
            # Low confidence - search multiple collections
            # Sort by score and take top collections
            sorted_collections = sorted(
                combined_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            selected_collections = []
            for collection, score in sorted_collections[:3]:  # Top 3
                if score > 0.1:  # Minimum threshold
                    selected_collections.append(collection)
            
            routing_strategy = "multi_collection"
            confidence = max_score
            
        else:
            # High confidence - route to best collection
            best_collection = max(combined_scores, key=combined_scores.get)
            selected_collections = [best_collection]
            routing_strategy = "single_collection"
            confidence = max_score
        
        # Special handling for question types
        if self._is_comparison_query(query):
            selected_collections = self.collection_names  # Search all for comparisons
            routing_strategy = "comparison_query"
        
        result = {
            "selected_collections": selected_collections,
            "routing_strategy": routing_strategy,
            "confidence": confidence,
            "scores": combined_scores,
            "rule_scores": rule_scores,
            "ml_scores": ml_scores,
            "reasoning": self._generate_routing_reasoning(
                query, selected_collections, routing_strategy, confidence
            )
        }
        
        logger.debug(f"Query routing: '{query}' -> {selected_collections} ({routing_strategy}, conf: {confidence:.2f})")
        
        return result
    
    def _is_comparison_query(self, query: str) -> bool:
        """Detect if query is asking for comparison"""
        comparison_patterns = [
            r"\b(?:compare|comparison|difference|vs|versus|against)\b",
            r"\b(?:better|best|worst|advantage|disadvantage)\b",
            r"\b(?:similar|different|alike|unlike)\b",
            r"\b(?:which|what.*between|how.*differ)\b"
        ]
        
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in comparison_patterns)
    
    def _generate_routing_reasoning(
        self, 
        query: str, 
        collections: List[str], 
        strategy: str, 
        confidence: float
    ) -> str:
        """Generate human-readable reasoning for routing decision"""
        
        if strategy == "single_collection":
            return f"High confidence ({confidence:.2f}) routing to '{collections[0]}' based on query content analysis"
        
        elif strategy == "multi_collection":
            collections_str = "', '".join(collections)
            return f"Medium confidence ({confidence:.2f}) - searching multiple collections: '{collections_str}'"
        
        elif strategy == "comparison_query":
            return "Detected comparison query - searching all collections for comprehensive results"
        
        else:
            return f"Using {strategy} with {len(collections)} collections"
    
    def update_classifier(self, new_examples: List[Tuple[str, str]]):
        """Update classifier with new training examples"""
        try:
            if not self.classifier:
                self._initialize_ml_classifier()
            
            if self.classifier and new_examples:
                # Get existing training data
                existing_data = self._generate_training_data()
                
                # Combine with new examples
                all_data = existing_data + new_examples
                
                # Retrain classifier
                texts, labels = zip(*all_data)
                self.classifier.fit(texts, labels)
                
                logger.info(f"Classifier updated with {len(new_examples)} new examples")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update classifier: {e}")
            return False
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get statistics about routing patterns"""
        return {
            "available_collections": self.collection_names,
            "routing_patterns_count": {
                collection: len(patterns) 
                for collection, patterns in self.routing_patterns.items()
            },
            "ml_classifier_trained": self.is_trained,
            "classifier_classes": list(self.classifier.classes_) if self.classifier else []
        }