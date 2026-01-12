"""
Evaluation Suite for Defect Triage Agent.

Evaluates the agent's performance on test datasets using various metrics.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from defect_triage_agent import DefectTriageAgent, TriageResult
from vector_memory import VectorMemory
from log_ingestor import LogIngestor

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EvaluationCase:
    """Represents a single test case for evaluation."""
    
    id: str
    error_log: str
    ground_truth_root_cause: str
    ground_truth_category: str
    severity: str = "medium"
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating triage performance."""
    
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_confidence: float
    confidence_calibration: float
    avg_response_time: float
    total_cases: int
    correct_predictions: int
    
    # Per-category metrics
    category_accuracy: Dict[str, float] = field(default_factory=dict)
    category_counts: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class EvaluationResult:
    """Result of evaluating a single case."""
    
    case_id: str
    predicted_root_cause: str
    ground_truth_root_cause: str
    predicted_category: str
    ground_truth_category: str
    confidence: float
    is_correct: bool
    response_time: float
    similar_defects_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DefectTriageEvaluator:
    """
    Evaluates the DefectTriageAgent on test datasets.
    
    Provides comprehensive metrics including accuracy, precision, recall,
    confidence calibration, and per-category performance.
    """
    
    def __init__(
        self,
        agent: DefectTriageAgent,
        category_keywords: Optional[Dict[str, List[str]]] = None
    ) -> None:
        """
        Initialize the evaluator.
        
        Args:
            agent: DefectTriageAgent instance to evaluate
            category_keywords: Keywords for categorizing root causes
        """
        logger.info("Initializing DefectTriageEvaluator")
        self.agent = agent
        
        # Default category keywords
        self.category_keywords = category_keywords or {
            "database": ["database", "connection", "sql", "timeout", "db"],
            "memory": ["memory", "heap", "outofmemory", "oom"],
            "null_pointer": ["null", "nullpointer", "npe"],
            "network": ["network", "socket", "connection refused", "unreachable"],
            "configuration": ["config", "configuration", "property", "setting"],
            "dependency": ["dependency", "library", "version", "compatibility"],
            "other": []
        }
    
    def _categorize_root_cause(self, root_cause: str) -> str:
        """
        Categorize a root cause based on keywords.
        
        Args:
            root_cause: Root cause description
            
        Returns:
            Category name
        """
        root_cause_lower = root_cause.lower()
        
        for category, keywords in self.category_keywords.items():
            if category == "other":
                continue
            
            if any(keyword in root_cause_lower for keyword in keywords):
                return category
        
        return "other"
    
    async def evaluate_single_case(
        self,
        case: EvaluationCase,
        top_k: int = 3
    ) -> EvaluationResult:
        """
        Evaluate the agent on a single test case.
        
        Args:
            case: Test case to evaluate
            top_k: Number of similar defects to retrieve
            
        Returns:
            EvaluationResult with metrics
        """
        logger.debug(f"Evaluating case: {case.id}")
        start_time = asyncio.get_event_loop().time()
        
        # Get agent prediction
        triage_result = await self.agent.analyze_defect(
            error_log=case.error_log,
            top_k=top_k
        )
        
        end_time = asyncio.get_event_loop().time()
        response_time = end_time - start_time
        
        # Categorize predictions
        predicted_category = self._categorize_root_cause(triage_result.root_cause)
        ground_truth_category = case.ground_truth_category
        
        # Check if prediction is correct (category-based)
        is_correct = predicted_category == ground_truth_category
        
        return EvaluationResult(
            case_id=case.id,
            predicted_root_cause=triage_result.root_cause,
            ground_truth_root_cause=case.ground_truth_root_cause,
            predicted_category=predicted_category,
            ground_truth_category=ground_truth_category,
            confidence=triage_result.confidence,
            is_correct=is_correct,
            response_time=response_time,
            similar_defects_count=len(triage_result.similar_defects)
        )
    
    async def evaluate_dataset(
        self,
        test_cases: List[EvaluationCase],
        top_k: int = 3
    ) -> tuple[List[EvaluationResult], EvaluationMetrics]:
        """
        Evaluate the agent on a complete test dataset.
        
        Args:
            test_cases: List of test cases
            top_k: Number of similar defects to retrieve
            
        Returns:
            Tuple of (results list, aggregated metrics)
        """
        logger.info(f"Evaluating agent on {len(test_cases)} test cases")
        print(f"🔬 Evaluating {len(test_cases)} test cases...\n")
        
        results: List[EvaluationResult] = []
        
        for i, case in enumerate(test_cases, 1):
            print(f"Evaluating case {i}/{len(test_cases)}: {case.id}")
            
            result = await self.evaluate_single_case(case, top_k=top_k)
            results.append(result)
            
            status = "✅" if result.is_correct else "❌"
            print(f"  {status} Predicted: {result.predicted_category} | "
                  f"Confidence: {result.confidence:.2%} | "
                  f"Time: {result.response_time:.2f}s\n")
        
        # Calculate metrics
        metrics = self._calculate_metrics(results, test_cases)
        
        return results, metrics
    
    def _calculate_metrics(
        self,
        results: List[EvaluationResult],
        test_cases: List[EvaluationCase]
    ) -> EvaluationMetrics:
        """
        Calculate aggregated evaluation metrics.
        
        Args:
            results: List of evaluation results
            test_cases: Original test cases
            
        Returns:
            EvaluationMetrics object
        """
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        
        # Overall accuracy
        accuracy = correct / total if total > 0 else 0.0
        
        # Get predictions and ground truth for sklearn metrics
        y_pred = [r.predicted_category for r in results]
        y_true = [r.ground_truth_category for r in results]
        
        # Calculate precision, recall, F1
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Average confidence
        avg_confidence = np.mean([r.confidence for r in results])
        
        # Confidence calibration (correlation between confidence and correctness)
        confidences = np.array([r.confidence for r in results])
        correctness = np.array([1.0 if r.is_correct else 0.0 for r in results])
        
        if len(confidences) > 1:
            confidence_calibration = np.corrcoef(confidences, correctness)[0, 1]
        else:
            confidence_calibration = 0.0
        
        # Average response time
        avg_response_time = np.mean([r.response_time for r in results])
        
        # Per-category metrics
        category_accuracy: Dict[str, float] = {}
        category_counts: Dict[str, int] = {}
        
        for category in set(y_true):
            category_results = [r for r in results if r.ground_truth_category == category]
            category_correct = sum(1 for r in category_results if r.is_correct)
            category_total = len(category_results)
            
            category_accuracy[category] = category_correct / category_total if category_total > 0 else 0.0
            category_counts[category] = category_total
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            avg_confidence=avg_confidence,
            confidence_calibration=confidence_calibration,
            avg_response_time=avg_response_time,
            total_cases=total,
            correct_predictions=correct,
            category_accuracy=category_accuracy,
            category_counts=category_counts
        )
    
    def generate_report(
        self,
        results: List[EvaluationResult],
        metrics: EvaluationMetrics,
        output_path: Optional[str | Path] = None
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: List of evaluation results
            metrics: Aggregated metrics
            output_path: Optional path to save the report
            
        Returns:
            Report as string
        """
        logger.info(f"Generating evaluation report for {len(results)} results")
        report = "=" * 80 + "\n"
        report += "DEFECT TRIAGE AGENT EVALUATION REPORT\n"
        report += "=" * 80 + "\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Overall metrics
        report += "-" * 80 + "\n"
        report += "OVERALL METRICS\n"
        report += "-" * 80 + "\n"
        report += f"Total Cases:           {metrics.total_cases}\n"
        report += f"Correct Predictions:   {metrics.correct_predictions}\n"
        report += f"Accuracy:              {metrics.accuracy:.2%}\n"
        report += f"Precision:             {metrics.precision:.2%}\n"
        report += f"Recall:                {metrics.recall:.2%}\n"
        report += f"F1 Score:              {metrics.f1_score:.2%}\n"
        report += f"Avg Confidence:        {metrics.avg_confidence:.2%}\n"
        report += f"Confidence Calibration: {metrics.confidence_calibration:.3f}\n"
        report += f"Avg Response Time:     {metrics.avg_response_time:.2f}s\n\n"
        
        # Per-category metrics
        report += "-" * 80 + "\n"
        report += "PER-CATEGORY PERFORMANCE\n"
        report += "-" * 80 + "\n"
        
        for category in sorted(metrics.category_accuracy.keys()):
            count = metrics.category_counts[category]
            acc = metrics.category_accuracy[category]
            report += f"{category.upper():20} | Count: {count:3} | Accuracy: {acc:.2%}\n"
        
        report += "\n"
        
        # Confidence distribution
        report += "-" * 80 + "\n"
        report += "CONFIDENCE DISTRIBUTION\n"
        report += "-" * 80 + "\n"
        
        confidences = [r.confidence for r in results]
        high_conf = sum(1 for c in confidences if c >= 0.7)
        medium_conf = sum(1 for c in confidences if 0.4 <= c < 0.7)
        low_conf = sum(1 for c in confidences if c < 0.4)
        
        report += f"High (≥0.7):    {high_conf} ({high_conf/len(results):.1%})\n"
        report += f"Medium (0.4-0.7): {medium_conf} ({medium_conf/len(results):.1%})\n"
        report += f"Low (<0.4):     {low_conf} ({low_conf/len(results):.1%})\n\n"
        
        # Sample incorrect predictions
        incorrect = [r for r in results if not r.is_correct]
        if incorrect:
            report += "-" * 80 + "\n"
            report += f"SAMPLE INCORRECT PREDICTIONS (showing {min(5, len(incorrect))})\n"
            report += "-" * 80 + "\n"
            
            for r in incorrect[:5]:
                report += f"Case ID: {r.case_id}\n"
                report += f"  Predicted: {r.predicted_category} (confidence: {r.confidence:.2%})\n"
                report += f"  Ground Truth: {r.ground_truth_category}\n\n"
        
        # Save to file if specified
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                f.write(report)
            
            print(f"📄 Report saved to: {path}")
        
        return report
    
    def export_results_to_csv(
        self,
        results: List[EvaluationResult],
        output_path: str | Path
    ) -> None:
        """
        Export evaluation results to CSV.
        
        Args:
            results: List of evaluation results
            output_path: Path to save CSV file
        logger.info(f"Exporting {len(results)} results to CSV: {output_path}")
        """
        df = pd.DataFrame([r.to_dict() for r in results])
        df.to_csv(output_path, index=False)
        print(f"📊 Results exported to: {output_path}")


def load_test_cases_from_json(file_path: str | Path) -> List[EvaluationCase]:
    """
    Load test cases from a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of EvaluationCase objects
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    cases = []
    for item in data:
        case = EvaluationCase(
            id=item["id"],
            error_log=item["error_log"],
            ground_truth_root_cause=item["ground_truth_root_cause"],
            ground_truth_category=item["ground_truth_category"],
            severity=item.get("severity", "medium"),
            tags=item.get("tags", [])
        )
        cases.append(case)
    
    return cases


def create_sample_test_dataset() -> List[EvaluationCase]:
    """
    Create a sample test dataset for demonstration.
    
    Returns:
        List of sample test cases
    """
    return [
        EvaluationCase(
            id="test_001",
            error_log="""
ERROR: Database connection pool exhausted
java.sql.SQLException: Cannot get connection, pool exhausted
at com.db.ConnectionPool.getConnection(ConnectionPool.java:89)
at com.service.UserService.findUser(UserService.java:34)
Caused by: Connection timeout after 30000ms
            """,
            ground_truth_root_cause="Database connection pool exhausted due to timeout",
            ground_truth_category="database",
            severity="high",
            tags=["database", "connection", "pool"]
        ),
        EvaluationCase(
            id="test_002",
            error_log="""
FATAL: Out of memory error
java.lang.OutOfMemoryError: Java heap space
at com.batch.DataProcessor.loadRecords(DataProcessor.java:156)
at com.batch.BatchJob.execute(BatchJob.java:234)
Heap dump created
            """,
            ground_truth_root_cause="Insufficient heap size for batch processing",
            ground_truth_category="memory",
            severity="critical",
            tags=["memory", "heap", "batch"]
        ),
        EvaluationCase(
            id="test_003",
            error_log="""
ERROR: Null pointer exception in payment flow
java.lang.NullPointerException: Cannot invoke 'getAmount' on null
at com.payment.PaymentService.processPayment(PaymentService.java:67)
at com.controller.PaymentController.checkout(PaymentController.java:89)
User session ID: abc123
            """,
            ground_truth_root_cause="Missing null check in payment processing",
            ground_truth_category="null_pointer",
            severity="high",
            tags=["nullpointer", "payment"]
        ),
    ]


async def main() -> None:
    """Example usage of the evaluation suite."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Check for Azure credentials
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if not all([azure_endpoint, azure_api_key, deployment_name]):
        print("❌ Error: Missing Azure OpenAI credentials")
        return
    
    # Setup vector memory with historical data
    vector_db_path = Path("./vector_db")
    
    if not vector_db_path.exists():
        print("❌ Error: Vector database not found. Run defect_triage_agent.py first.")
        return
    
    print("📚 Loading vector database...")
    memory = VectorMemory.load(vector_db_path)
    
    # Create agent
    print("🤖 Initializing agent...")
    agent = DefectTriageAgent(
        vector_memory=memory,
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        deployment_name=deployment_name,
        temperature=0.3
    )
    
    # Create evaluator
    evaluator = DefectTriageEvaluator(agent)
    
    # Create sample test dataset
    print("\n📋 Creating sample test dataset...")
    test_cases = create_sample_test_dataset()
    
    # Run evaluation
    results, metrics = await evaluator.evaluate_dataset(test_cases, top_k=3)
    
    # Generate report
    print("\n" + "=" * 80)
    report = evaluator.generate_report(
        results,
        metrics,
        output_path="evaluation_report.txt"
    )
    print(report)
    
    # Export results
    evaluator.export_results_to_csv(results, "evaluation_results.csv")
    
    # Save metrics as JSON
    with open("evaluation_metrics.json", 'w') as f:
        f.write(metrics.to_json())
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())
