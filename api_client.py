"""
Example client for interacting with the Defect Triage API.

Demonstrates how to use the REST API endpoints.
"""

import requests
import json
from typing import List, Dict, Any, Optional


class DefectTriageClient:
    """Client for the Defect Triage API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        response = self.session.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def analyze_defect(
        self,
        error_log: str,
        top_k: int = 3,
        include_similar: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a single error log.
        
        Args:
            error_log: The error log text
            top_k: Number of similar defects to retrieve
            include_similar: Include similar defects in response
            
        Returns:
            Triage result with root cause and recommendations
        """
        payload = {
            "error_log": error_log,
            "top_k": top_k,
            "include_similar": include_similar
        }
        
        response = self.session.post(
            f"{self.base_url}/analyze",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def batch_analyze(
        self,
        error_logs: List[str],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze multiple error logs in batch.
        
        Args:
            error_logs: List of error log texts
            top_k: Number of similar defects per error
            
        Returns:
            Batch triage results
        """
        payload = {
            "error_logs": error_logs,
            "top_k": top_k
        }
        
        response = self.session.post(
            f"{self.base_url}/analyze/batch",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def add_defect(
        self,
        error_log: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a new defect to the knowledge base.
        
        Args:
            error_log: The error log text
            metadata: Optional metadata
            source: Source system
            
        Returns:
            Confirmation response
        """
        payload = {
            "error_log": error_log,
            "metadata": metadata or {},
            "source": source
        }
        
        response = self.session.post(
            f"{self.base_url}/add-defect",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def clear_knowledge_base(self, confirm: bool = False) -> Dict[str, Any]:
        """
        Clear the knowledge base.
        
        Args:
            confirm: Must be True to proceed
            
        Returns:
            Confirmation response
        """
        response = self.session.post(
            f"{self.base_url}/clear-knowledge-base",
            params={"confirm": confirm}
        )
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the client."""
    # Initialize client
    client = DefectTriageClient("http://localhost:8000")
    
    print("🔍 Defect Triage API Client Example\n")
    
    # Health check
    print("1. Health Check:")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Vector DB Size: {health['vector_db_size']} defects")
    print(f"   Uptime: {health['uptime_seconds']:.2f}s\n")
    
    # Get stats
    print("2. System Statistics:")
    stats = client.get_stats()
    print(f"   Total Defects: {stats['total_defects']}")
    print(f"   Model: {stats['model_name']}")
    print(f"   Index Type: {stats['index_type']}\n")
    
    # Example error log
    error_log = """
    ERROR: Database connection timeout
    java.sql.SQLException: Connection timed out after 30 seconds
    at com.database.ConnectionManager.connect(ConnectionManager.java:89)
    at com.service.ProductService.getProducts(ProductService.java:34)
    Connection pool exhausted
    """
    
    # Analyze single defect
    print("3. Analyzing Single Defect:")
    result = client.analyze_defect(error_log, top_k=3)
    print(f"   Root Cause: {result['root_cause']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Analysis Time: {result['analysis_time_ms']:.2f}ms")
    print(f"   Recommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"     {i}. {rec}")
    print()
    
    # Batch analysis
    print("4. Batch Analysis:")
    batch_logs = [
        "ERROR: NullPointerException in payment service",
        "FATAL: OutOfMemoryError during batch processing"
    ]
    
    batch_result = client.batch_analyze(batch_logs, top_k=3)
    print(f"   Analyzed: {batch_result['total_count']} errors")
    print(f"   Total Time: {batch_result['total_time_ms']:.2f}ms\n")
    
    # Add new defect
    print("5. Adding New Defect to Knowledge Base:")
    new_defect = """
    ERROR: Configuration error
    java.lang.IllegalStateException: Required property 'api.key' not found
    at com.config.ConfigLoader.validate(ConfigLoader.java:45)
    """
    
    add_result = client.add_defect(
        new_defect,
        metadata={"severity": "high", "team": "platform"},
        source="Jenkins"
    )
    print(f"   Chunks Added: {add_result['chunks_added']}")
    print(f"   Total Defects: {add_result['total_defects']}")
    print(f"   Message: {add_result['message']}\n")
    
    print("✅ Client example completed!")


if __name__ == "__main__":
    main()
