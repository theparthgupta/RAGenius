"""
Analytics and monitoring system for the Advanced RAG System
Tracks queries, performance metrics, and system insights
"""
import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from .config import DB_CONFIG

logger = logging.getLogger(__name__)

class QueryAnalytics:
    """Advanced analytics for query tracking and system monitoring"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_CONFIG.ANALYTICS_DB_PATH
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Queries table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS queries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        query_text TEXT NOT NULL,
                        query_type TEXT,
                        selected_collections TEXT,
                        routing_confidence REAL,
                        response_time REAL,
                        context_docs_count INTEGER,
                        response_length INTEGER,
                        model_used TEXT,
                        user_rating INTEGER,
                        success BOOLEAN DEFAULT 1,
                        error_message TEXT
                    )
                """)
                
                # Performance metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metric_type TEXT NOT NULL,
                        metric_value REAL,
                        metadata TEXT
                    )
                """)
                
                # System events table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        event_type TEXT NOT NULL,
                        event_description TEXT,
                        severity TEXT DEFAULT 'INFO',
                        metadata TEXT
                    )
                """)
                
                # Collections statistics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS collection_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        collection_name TEXT NOT NULL,
                        document_count INTEGER,
                        query_count INTEGER DEFAULT 0,
                        avg_response_time REAL,
                        success_rate REAL
                    )
                """)
                
                conn.commit()
                logger.info("Analytics database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing analytics database: {e}")
            raise
    
    def log_query(
        self,
        query: str,
        routing_result: Dict[str, Any],
        llm_result: Dict[str, Any],
        user_rating: Optional[int] = None
    ):
        """Log query and response details"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO queries (
                        query_text, query_type, selected_collections, routing_confidence,
                        response_time, context_docs_count, response_length, model_used,
                        user_rating, success, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query,
                    routing_result.get('routing_strategy', 'unknown'),
                    json.dumps(routing_result.get('selected_collections', [])),
                    routing_result.get('confidence', 0.0),
                    llm_result.get('generation_time', 0.0),
                    llm_result.get('context_used', 0),
                    llm_result.get('response_length', 0),
                    llm_result.get('model_used', 'unknown'),
                    user_rating,
                    not bool(llm_result.get('error')),
                    llm_result.get('error')
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error logging query: {e}")
    
    def log_performance_metric(self, metric_type: str, value: float, metadata: Dict[str, Any] = None):
        """Log performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO performance_metrics (metric_type, metric_value, metadata)
                    VALUES (?, ?, ?)
                """, (
                    metric_type,
                    value,
                    json.dumps(metadata or {})
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error logging performance metric: {e}")
    
    def log_system_event(self, event_type: str, description: str, severity: str = "INFO", metadata: Dict[str, Any] = None):
        """Log system events"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO system_events (event_type, event_description, severity, metadata)
                    VALUES (?, ?, ?, ?)
                """, (
                    event_type,
                    description,
                    severity,
                    json.dumps(metadata or {})
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error logging system event: {e}")
    
    def update_collection_stats(self, collection_name: str, document_count: int):
        """Update collection statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Calculate current metrics
                cursor.execute("""
                    SELECT COUNT(*), AVG(response_time), AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END)
                    FROM queries 
                    WHERE selected_collections LIKE ?
                    AND timestamp >= datetime('now', '-7 days')
                """, (f'%{collection_name}%',))
                
                result = cursor.fetchone()
                query_count, avg_response_time, success_rate = result or (0, 0.0, 1.0)
                
                # Insert or update stats
                cursor.execute("""
                    INSERT OR REPLACE INTO collection_stats 
                    (collection_name, document_count, query_count, avg_response_time, success_rate)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    collection_name,
                    document_count,
                    query_count or 0,
                    avg_response_time or 0.0,
                    success_rate or 1.0
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating collection stats: {e}")
    
    def get_query_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive query analytics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Basic query statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_queries,
                        AVG(response_time) as avg_response_time,
                        AVG(routing_confidence) as avg_confidence,
                        AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                        AVG(context_docs_count) as avg_context_docs
                    FROM queries 
                    WHERE timestamp >= datetime('now', '-{} days')
                """.format(days))
                
                basic_stats = cursor.fetchone()
                
                # Query frequency by hour
                cursor.execute("""
                    SELECT 
                        strftime('%H', timestamp) as hour,
                        COUNT(*) as query_count
                    FROM queries 
                    WHERE timestamp >= datetime('now', '-{} days')
                    GROUP BY strftime('%H', timestamp)
                    ORDER BY hour
                """.format(days))
                
                hourly_data = cursor.fetchall()
                
                # Collection usage statistics
                cursor.execute("""
                    SELECT 
                        query_type,
                        COUNT(*) as count,
                        AVG(response_time) as avg_time,
                        AVG(routing_confidence) as avg_confidence
                    FROM queries 
                    WHERE timestamp >= datetime('now', '-{} days')
                    GROUP BY query_type
                    ORDER BY count DESC
                """.format(days))
                
                collection_usage = cursor.fetchall()
                
                # Response time distribution
                cursor.execute("""
                    SELECT 
                        CASE 
                            WHEN response_time < 1.0 THEN 'Under 1s'
                            WHEN response_time < 2.0 THEN '1-2s'
                            WHEN response_time < 5.0 THEN '2-5s'
                            ELSE 'Over 5s'
                        END as time_bucket,
                        COUNT(*) as count
                    FROM queries 
                    WHERE timestamp >= datetime('now', '-{} days')
                    GROUP BY time_bucket
                """.format(days))
                
                response_time_dist = cursor.fetchall()
                
                # Model usage statistics
                cursor.execute("""
                    SELECT 
                        model_used,
                        COUNT(*) as usage_count,
                        AVG(response_time) as avg_response_time
                    FROM queries 
                    WHERE timestamp >= datetime('now', '-{} days')
                    GROUP BY model_used
                """.format(days))
                
                model_usage = cursor.fetchall()
                
                return {
                    "basic_stats": {
                        "total_queries": basic_stats[0] or 0,
                        "avg_response_time": round(basic_stats[1] or 0, 2),
                        "avg_confidence": round(basic_stats[2] or 0, 2),
                        "success_rate": round(basic_stats[3] or 0, 2),
                        "avg_context_docs": round(basic_stats[4] or 0, 1)
                    },
                    "hourly_distribution": dict(hourly_data),
                    "collection_usage": [
                        {
                            "type": row[0],
                            "count": row[1],
                            "avg_time": round(row[2], 2),
                            "avg_confidence": round(row[3], 2)
                        }
                        for row in collection_usage
                    ],
                    "response_time_distribution": dict(response_time_dist),
                    "model_usage": [
                        {
                            "model": row[0],
                            "count": row[1],
                            "avg_time": round(row[2], 2)
                        }
                        for row in model_usage
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error getting query analytics: {e}")
            return {}
    
    def get_performance_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get performance trends over time"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Daily performance trends
                df_queries = pd.read_sql_query(f"""
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as query_count,
                        AVG(response_time) as avg_response_time,
                        AVG(routing_confidence) as avg_confidence,
                        SUM(CASE WHEN success THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
                    FROM queries 
                    WHERE timestamp >= datetime('now', '-{days} days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """, conn)
                
                # Performance metrics trends
                df_metrics = pd.read_sql_query(f"""
                    SELECT 
                        DATE(timestamp) as date,
                        metric_type,
                        AVG(metric_value) as avg_value
                    FROM performance_metrics
                    WHERE timestamp >= datetime('now', '-{days} days')
                    GROUP BY DATE(timestamp), metric_type
                    ORDER BY date
                """, conn)
                
                return {
                    "daily_trends": df_queries.to_dict('records'),
                    "metric_trends": df_metrics.to_dict('records')
                }
                
        except Exception as e:
            logger.error(f"Error getting performance trends: {e}")
            return {"daily_trends": [], "metric_trends": []}
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Recent error rate
                cursor.execute("""
                    SELECT 
                        AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                        COUNT(*) as total_queries
                    FROM queries 
                    WHERE timestamp >= datetime('now', '-1 hour')
                """)
                
                recent_stats = cursor.fetchone()
                success_rate = recent_stats[0] if recent_stats[0] is not None else 1.0
                
                # System events in last 24 hours
                cursor.execute("""
                    SELECT severity, COUNT(*) as count
                    FROM system_events 
                    WHERE timestamp >= datetime('now', '-24 hours')
                    GROUP BY severity
                """)
                
                events = dict(cursor.fetchall())
                
                # Average response time trend
                cursor.execute("""
                    SELECT AVG(response_time)
                    FROM queries 
                    WHERE timestamp >= datetime('now', '-1 hour')
                """)
                
                recent_avg_time = cursor.fetchone()[0] or 0
                
                # Health status determination
                if success_rate >= 0.95 and recent_avg_time < 3.0 and events.get('ERROR', 0) == 0:
                    health_status = "Excellent"
                elif success_rate >= 0.90 and recent_avg_time < 5.0 and events.get('ERROR', 0) < 3:
                    health_status = "Good"
                elif success_rate >= 0.80:
                    health_status = "Fair"
                else:
                    health_status = "Poor"
                
                return {
                    "health_status": health_status,
                    "success_rate": round(success_rate, 3),
                    "avg_response_time": round(recent_avg_time, 2),
                    "recent_events": events,
                    "total_recent_queries": recent_stats[1] or 0
                }
                
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                "health_status": "Unknown",
                "success_rate": 0,
                "avg_response_time": 0,
                "recent_events": {},
                "total_recent_queries": 0
            }
    
    def get_user_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of user feedback and ratings"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        AVG(user_rating) as avg_rating,
                        COUNT(CASE WHEN user_rating IS NOT NULL THEN 1 END) as total_ratings,
                        COUNT(CASE WHEN user_rating >= 4 THEN 1 END) as positive_ratings,
                        COUNT(CASE WHEN user_rating <= 2 THEN 1 END) as negative_ratings
                    FROM queries 
                    WHERE timestamp >= datetime('now', '-30 days')
                """)
                
                feedback_stats = cursor.fetchone()
                
                if feedback_stats and feedback_stats[1] > 0:  # If there are ratings
                    avg_rating = round(feedback_stats[0], 2)
                    total_ratings = feedback_stats[1]
                    satisfaction_rate = round((feedback_stats[2] / total_ratings) * 100, 1)
                else:
                    avg_rating = 0
                    total_ratings = 0
                    satisfaction_rate = 0
                
                return {
                    "avg_rating": avg_rating,
                    "total_ratings": total_ratings,
                    "satisfaction_rate": satisfaction_rate,
                    "positive_ratings": feedback_stats[2] if feedback_stats else 0,
                    "negative_ratings": feedback_stats[3] if feedback_stats else 0
                }
                
        except Exception as e:
            logger.error(f"Error getting user feedback summary: {e}")
            return {
                "avg_rating": 0,
                "total_ratings": 0,
                "satisfaction_rate": 0,
                "positive_ratings": 0,
                "negative_ratings": 0
            }
    
    def generate_insights(self) -> List[str]:
        """Generate actionable insights from analytics data"""
        insights = []
        
        try:
            analytics = self.get_query_analytics(days=7)
            health = self.get_system_health()
            
            basic_stats = analytics.get("basic_stats", {})
            
            # Response time insights
            avg_time = basic_stats.get("avg_response_time", 0)
            if avg_time > 3.0:
                insights.append(f"⚠️ Average response time is {avg_time}s - consider model optimization")
            elif avg_time < 1.0:
                insights.append(f"✅ Excellent response time: {avg_time}s")
            
            # Success rate insights
            success_rate = basic_stats.get("success_rate", 0)
            if success_rate < 0.9:
                insights.append(f"⚠️ Success rate is {success_rate*100:.1f}% - investigate error patterns")
            
            # Confidence insights
            avg_confidence = basic_stats.get("avg_confidence", 0)
            if avg_confidence < 0.7:
                insights.append(f"💡 Low routing confidence ({avg_confidence:.2f}) - consider adding more training data")
            
            # Usage pattern insights
            collection_usage = analytics.get("collection_usage", [])
            if collection_usage:
                most_used = max(collection_usage, key=lambda x: x["count"])
                insights.append(f"📊 Most queried collection: {most_used['type']} ({most_used['count']} queries)")
            
            # System health insights
            if health.get("health_status") == "Poor":
                insights.append("🚨 System health is poor - immediate attention required")
            elif health.get("health_status") == "Excellent":
                insights.append("🎉 System running at peak performance")
            
            # Query volume insights
            total_queries = basic_stats.get("total_queries", 0)
            if total_queries > 100:
                insights.append(f"📈 High query volume: {total_queries} queries in last 7 days")
            elif total_queries < 10:
                insights.append("📉 Low query volume - system may need promotion")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights.append("⚠️ Unable to generate insights due to data access issues")
        
        return insights if insights else ["✅ System operating normally with no significant issues detected"]
    
    def export_analytics_data(self, days: int = 30) -> Dict[str, Any]:
        """Export analytics data for external analysis"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Export queries data
                df_queries = pd.read_sql_query(f"""
                    SELECT * FROM queries 
                    WHERE timestamp >= datetime('now', '-{days} days')
                    ORDER BY timestamp DESC
                """, conn)
                
                # Export performance metrics
                df_metrics = pd.read_sql_query(f"""
                    SELECT * FROM performance_metrics 
                    WHERE timestamp >= datetime('now', '-{days} days')
                    ORDER BY timestamp DESC
                """, conn)
                
                # Export system events
                df_events = pd.read_sql_query(f"""
                    SELECT * FROM system_events 
                    WHERE timestamp >= datetime('now', '-{days} days')
                    ORDER BY timestamp DESC
                """, conn)
                
                return {
                    "queries": df_queries.to_dict('records'),
                    "metrics": df_metrics.to_dict('records'),
                    "events": df_events.to_dict('records'),
                    "export_timestamp": datetime.now().isoformat(),
                    "days_included": days
                }
                
        except Exception as e:
            logger.error(f"Error exporting analytics data: {e}")
            return {}