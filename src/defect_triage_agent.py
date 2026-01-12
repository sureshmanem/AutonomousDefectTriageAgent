"""
Defect Triage Agent using Microsoft Semantic Kernel and Azure OpenAI.

This module provides an intelligent agent that analyzes Jenkins failure logs
by comparing them against historical defects using RAG (Retrieval-Augmented Generation).
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

from vector_memory import VectorMemory, SearchResult
from log_ingestor import LogIngestor

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TriageResult:
    """Result of defect triage analysis."""
    
    root_cause: str
    confidence: float
    similar_defects: List[Dict[str, Any]]
    reasoning: str
    recommendations: List[str]
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DefectTriageAgent:
    """
    Semantic Kernel-based agent for automated defect triage.
    
    Uses RAG pattern to retrieve similar historical defects and
    Azure OpenAI to analyze and provide root cause analysis.
    """
    
    def __init__(
        self,
        vector_memory: VectorMemory,
        azure_endpoint: str,
        azure_api_key: str,
        deployment_name: str,
        api_version: str = "2024-02-01",
        temperature: float = 0.3,
        max_tokens: int = 1500
    ) -> None:
        """
        Initialize the Defect Triage Agent.
        
        Args:
            vector_memory: VectorMemory instance with historical defects
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_key: Azure OpenAI API key
            deployment_name: Name of the deployed model
            api_version: Azure OpenAI API version
            temperature: LLM temperature (0-1, lower = more deterministic)
            max_tokens: Maximum tokens for LLM response
        """
        logger.info(f"Initializing DefectTriageAgent with deployment={deployment_name}, temperature={temperature}")
        self.vector_memory = vector_memory
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Semantic Kernel
        self.kernel = Kernel()
        
        # Add Azure OpenAI service
        self.service_id = "azure_openai_chat"
        self.kernel.add_service(
            AzureChatCompletion(
                service_id=self.service_id,
                deployment_name=deployment_name,
                endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=api_version,
            )
        )
        
        # Log ingestor for processing new logs
        self.log_ingestor = LogIngestor()
    
    def _create_analysis_prompt(
        self,
        new_error: str,
        similar_defects: List[SearchResult]
    ) -> str:
        """
        Create the prompt for root cause analysis.
        
        Args:
            new_error: The new error log to analyze
            similar_defects: List of similar historical defects
            
        Returns:
            Formatted prompt string
        """
        logger.debug(f"Creating analysis prompt with {len(similar_defects)} similar defects")
        prompt = """You are an expert DevOps engineer specializing in defect triage and root cause analysis.

**Task**: Analyze a new error log by comparing it with similar historical defects and provide a detailed root cause analysis.

**New Error Log**:
```
{new_error}
```

**Similar Historical Defects**:

{historical_defects}

**Instructions**:
1. Carefully compare the new error with each historical defect
2. Identify common patterns, stack traces, and error types
3. Determine the most likely root cause based on similarities
4. Provide a confidence score (0.0 to 1.0) based on:
   - Similarity of error messages
   - Matching stack traces
   - Common failure patterns
   - Consistency across historical defects
5. Suggest actionable recommendations for resolution

**Output Format** (JSON only, no markdown):
{{
  "root_cause": "Clear description of the root cause",
  "confidence": 0.85,
  "reasoning": "Detailed explanation of your analysis",
  "recommendations": [
    "Specific action 1",
    "Specific action 2"
  ]
}}

**Important**: 
- Only output valid JSON
- Confidence must be between 0.0 and 1.0
- Be specific and actionable in recommendations
- If no clear pattern, confidence should be low (< 0.5)
"""
        
        # Format historical defects
        historical_section = ""
        for i, result in enumerate(similar_defects, 1):
            historical_section += f"\n**Historical Defect #{i}** (Similarity Score: {result.score:.4f}):\n"
            historical_section += f"```\n{result.chunk.content[:500]}\n```\n"
        
        return prompt.format(
            new_error=new_error[:1000],  # Limit new error length
            historical_defects=historical_section
        )
    
    async def analyze_defect(
        self,
        error_log: str,
        top_k: int = 3,
        score_threshold: Optional[float] = None
    ) -> TriageResult:
        """
        Analyze a new error log and provide root cause analysis.
        
        Args:
            error_log: The new error log string
            top_k: Number of similar historical defects to retrieve
            score_threshold: Minimum similarity score threshold
            
        Returns:
            TriageResult with analysis and recommendations
        """
        logger.info(f"Analyzing defect with top_k={top_k}, error_log_length={len(error_log)}")
        # Step 1: Search for similar historical defects
        print("🔍 Searching for similar historical defects...")
        similar_defects = await self.vector_memory.search_similar_async(
            query_text=error_log,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        if not similar_defects:
            return TriageResult(
                root_cause="No similar historical defects found",
                confidence=0.0,
                similar_defects=[],
                reasoning="Insufficient historical data for comparison",
                recommendations=["Add this defect to knowledge base", "Manual investigation required"]
            )
        
        print(f"✅ Found {len(similar_defects)} similar defects")
        
        # Step 2: Create analysis prompt
        prompt = self._create_analysis_prompt(error_log, similar_defects)
        
        # Step 3: Get LLM analysis
        print("🤖 Analyzing with Azure OpenAI...")
        
        # Create chat history
        chat_history = ChatHistory()
        chat_history.add_user_message(prompt)
        
        # Set execution settings
        execution_settings = AzureChatPromptExecutionSettings(
            service_id=self.service_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"}  # Force JSON response
        )
        
        # Get chat completion service
        chat_service = self.kernel.get_service(type=AzureChatCompletion)
        
        # Invoke LLM
        response = await chat_service.get_chat_message_contents(
            chat_history=chat_history,
            settings=execution_settings,
        )
        
        # Extract response
        llm_response = str(response[0].content)
        
        print("✅ Analysis complete")
        
        # Step 4: Parse LLM response
        try:
            analysis = json.loads(llm_response)
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse LLM response: {e}")
            # Fallback response
            analysis = {
                "root_cause": "Analysis failed - invalid response format",
                "confidence": 0.0,
                "reasoning": f"LLM response parsing error: {str(e)}",
                "recommendations": ["Retry analysis", "Check LLM configuration"]
            }
        
        # Step 5: Build result
        similar_defects_data = [
            {
                "score": result.score,
                "content_preview": result.chunk.content[:200],
                "line_range": f"{result.chunk.line_start}-{result.chunk.line_end}",
                "metadata": result.metadata
            }
            for result in similar_defects
        ]
        
        return TriageResult(
            root_cause=analysis.get("root_cause", "Unknown"),
            confidence=float(analysis.get("confidence", 0.0)),
            similar_defects=similar_defects_data,
            reasoning=analysis.get("reasoning", "No reasoning provided"),
            recommendations=analysis.get("recommendations", [])
        )
    
    async def analyze_log_file(
        self,
        log_file_path: str | Path,
        top_k: int = 3
    ) -> List[TriageResult]:
        """
        Analyze a log file and triage all errors found.
        
        Args:
            log_file_path: Path to the log file
            top_k: Number of similar defects to retrieve per error
            
        Returns:
            List of TriageResult for each error chunk
        """
        logger.info(f"Analyzing log file: {log_file_path}")
        # Process log file
        print(f"📄 Processing log file: {log_file_path}")
        chunks = self.log_ingestor.process_log_file(log_file_path)
        
        print(f"📦 Found {len(chunks)} error chunks")
        
        # Analyze each chunk
        results: List[TriageResult] = []
        for i, chunk in enumerate(chunks, 1):
            print(f"\n--- Analyzing chunk {i}/{len(chunks)} ---")
            result = await self.analyze_defect(chunk.content, top_k=top_k)
            results.append(result)
        
        return results
    
    async def batch_analyze(
        self,
        error_logs: List[str],
        top_k: int = 3
    ) -> List[TriageResult]:
        """
        Analyze multiple error logs in batch.
        
        Args:
            error_logs: List of error log strings
            top_k: Number of similar defects per error
            
        Returns:
            List of TriageResult
        """
        logger.info(f"Batch analyzing {len(error_logs)} error logs")
        results: List[TriageResult] = []
        
        for i, error_log in enumerate(error_logs, 1):
            print(f"\n--- Analyzing error {i}/{len(error_logs)} ---")
            result = await self.analyze_defect(error_log, top_k=top_k)
            results.append(result)
        
        return results
    
    def get_summary_report(self, results: List[TriageResult]) -> str:
        """
        Generate a summary report of multiple triage results.
        
        Args:
            results: List of TriageResult objects
            
        Returns:
            Formatted summary report
        """
        logger.info(f"Generating summary report for {len(results)} results")
        report = "=" * 80 + "\n"
        report += "DEFECT TRIAGE SUMMARY REPORT\n"
        report += "=" * 80 + "\n\n"
        
        total = len(results)
        high_confidence = sum(1 for r in results if r.confidence >= 0.7)
        medium_confidence = sum(1 for r in results if 0.4 <= r.confidence < 0.7)
        low_confidence = sum(1 for r in results if r.confidence < 0.4)
        
        report += f"Total Defects Analyzed: {total}\n"
        report += f"High Confidence (≥0.7): {high_confidence}\n"
        report += f"Medium Confidence (0.4-0.7): {medium_confidence}\n"
        report += f"Low Confidence (<0.4): {low_confidence}\n\n"
        
        report += "-" * 80 + "\n"
        report += "DETAILED RESULTS\n"
        report += "-" * 80 + "\n\n"
        
        for i, result in enumerate(results, 1):
            report += f"Defect #{i}:\n"
            report += f"  Root Cause: {result.root_cause}\n"
            report += f"  Confidence: {result.confidence:.2%}\n"
            report += f"  Similar Defects Found: {len(result.similar_defects)}\n"
            report += f"  Recommendations: {', '.join(result.recommendations[:2])}\n\n"
        
        return report


async def main() -> None:
    """Example usage of DefectTriageAgent."""
    import asyncio
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get Azure OpenAI credentials
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if not all([azure_endpoint, azure_api_key, deployment_name]):
        print("❌ Error: Missing Azure OpenAI credentials in .env file")
        print("Please set: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME")
        return
    
    # Load or create vector memory
    vector_db_path = Path("./vector_db")
    
    if vector_db_path.exists():
        print("📚 Loading existing vector database...")
        memory = VectorMemory.load(vector_db_path)
    else:
        print("📚 Creating new vector database with sample data...")
        from log_ingestor import LogIngestor
        
        # Sample historical defects
        historical_logs = [
            """
            ERROR: Database connection failed
            java.sql.SQLException: Connection timeout after 30000ms
            at com.db.ConnectionPool.getConnection(ConnectionPool.java:123)
            at com.service.UserService.findUser(UserService.java:45)
            Root cause: Database server not responding
            Resolution: Restarted database server
            """,
            """
            NullPointerException in payment processing
            java.lang.NullPointerException: Cannot invoke 'getAmount' on null object
            at com.payment.PaymentService.processPayment(PaymentService.java:67)
            at com.controller.PaymentController.checkout(PaymentController.java:89)
            Root cause: Missing payment validation
            Resolution: Added null check before processing
            """,
            """
            OutOfMemoryError during batch job
            java.lang.OutOfMemoryError: Java heap space
            at com.batch.DataProcessor.loadRecords(DataProcessor.java:156)
            at com.batch.BatchJob.execute(BatchJob.java:234)
            Root cause: Insufficient heap size for large dataset
            Resolution: Increased heap from 2GB to 4GB
            """
        ]
        
        ingestor = LogIngestor()
        chunks = []
        for log in historical_logs:
            chunks.extend(ingestor.process_log_string(log))
        
        memory = VectorMemory()
        await memory.add_documents_async(chunks)
        memory.save(vector_db_path)
    
    # Create agent
    print("\n🤖 Initializing Defect Triage Agent...")
    agent = DefectTriageAgent(
        vector_memory=memory,
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        deployment_name=deployment_name,
        temperature=0.3
    )
    
    # New error to analyze
    new_error = """
    ERROR: Failed to connect to database
    java.sql.SQLException: Connection timed out after 30 seconds
    at com.database.ConnectionManager.connect(ConnectionManager.java:89)
    at com.service.ProductService.getProducts(ProductService.java:34)
    Stack trace indicates network connectivity issue
    """
    
    print("\n" + "=" * 80)
    print("ANALYZING NEW DEFECT")
    print("=" * 80)
    print(f"\n{new_error}\n")
    
    # Analyze
    result = await agent.analyze_defect(new_error, top_k=3)
    
    # Display results
    print("\n" + "=" * 80)
    print("TRIAGE RESULTS")
    print("=" * 80)
    print(result.to_json())
    
    print("\n📊 Summary Report:")
    print(agent.get_summary_report([result]))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
