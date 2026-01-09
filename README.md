# Autonomous Defect Triage Agent

An Agentic AI system that automates the analysis of Jenkins failure logs using Microsoft Semantic Kernel and Azure OpenAI.

## Architecture

### 1. Log Ingestor (`log_ingestor.py`) ✅
- Reads Jenkins failure logs
- Removes timestamps using regex patterns
- Chunks text into 50-line blocks centered around Exception keywords
- Uses strict type hinting and Python 3.10+ syntax

### 2. Vector Memory (`vector_memory.py`) ✅
- FAISS-based vector storage with multiple index types
- Sentence-transformers embeddings (all-MiniLM-L6-v2)
- Methods: `add_documents()`, `search_similar()`, `save()`, `load()`
- Full async/await support for non-blocking operations

### 3. Semantic Kernel Agent (`defect_triage_agent.py`) ✅
- Takes new error logs as input
- RAG pattern: Retrieves top 3 similar historical defects
- Azure OpenAI analysis with structured JSON output
- Returns root cause, confidence score, and recommendations
- Full async/await implementation

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your Azure OpenAI credentials:
```bash
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
```

## Usage

### Step 1: Process Log Files

```python
from log_ingestor import LogIngestor

# Initialize ingestor
ingestor = LogIngestor(chunk_size=50)

# Process a log file
chunks = ingestor.process_log_file("jenkins_build_failure.log")

# Or process a string
log_content = "..."
chunks = ingestor.process_log_string(log_content)

# Convert to DataFrame
df = ingestor.chunks_to_dataframe(chunks)
```

### Step 2: Build Vector Memory

```python
from vector_memory import VectorMemory
import asyncio

async def setup_vector_db():
    # Initialize vector memory
    memory = VectorMemory(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        index_type="flat"  # or "ivf", "hnsw" for larger datasets
    )
    
    # Add historical defect logs
    await memory.add_documents_async(chunks)
    
    # Search for similar defects
    query = "NullPointerException in payment processing"
    results = await memory.search_similar_async(query, top_k=3)
    
    for result in results:
        print(f"Score: {result.score:.4f}")
        print(f"Content: {result.chunk.content[:100]}...")
    
    # Save to disk
    memory.save("./vector_db")
    
    # Load later
    loaded_memory = VectorMemory.load("./vector_db")

asyncio.run(setup_vector_db())
```

### Step 3: Run Defect Triage Agent

```python
from defect_triage_agent import DefectTriageAgent
from vector_memory import VectorMemory
import asyncio
import os
from dotenv import load_dotenv

async def triage_defect():
    # Load credentials
    load_dotenv()
    
    # Load vector memory
    memory = VectorMemory.load("./vector_db")
    
    # Create agent
    agent = DefectTriageAgent(
        vector_memory=memory,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    )
    
    # Analyze new error
    new_error = """
    ERROR: Database connection timeout
    java.sql.SQLException: Connection timed out
    """
    
    result = await agent.analyze_defect(new_error, top_k=3)
    
    # View results
    print(result.to_json())
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Root Cause: {result.root_cause}")

asyncio.run(triage_defect())
```

## Quick Test

Run the complete pipeline:

```bash
# Test log ingestion
python log_ingestor.py

# Test vector memory
python vector_memory.py

# Test defect triage agent (requires Azure OpenAI)
python defect_triage_agent.py
```

## Complete Example

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# 2. Run end-to-end triage
python defect_triage_agent.py
```

## Features

- ✅ Comprehensive timestamp removal (ISO, Unix, custom formats)
- ✅ Intelligent chunking around multiple exception types
- ✅ Overlap prevention for duplicate chunks
- ✅ Type-safe with Python 3.10+ type hints
- ✅ Pandas integration for data manipulation
- ✅ Extensible keyword matching
- ✅ FAISS vector search (flat, IVF, HNSW indexes)
- ✅ Async/await for non-blocking operations
- ✅ Persistent storage (save/load from disk)
- ✅ L2 distance similarity scoring
- ✅ RAG pattern with Semantic Kernel
- ✅ Structured JSON output from LLM
- ✅ Confidence scoring and recommendations
- ✅ Batch analysis support
- ✅ Comprehensive unit tests with pytest
- ✅ Evaluation suite with multiple metrics
- ✅ Per-category performance analysis
- ✅ Confidence calibration measurement

## Output Format

The agent returns structured JSON output:

```json
{
  "root_cause": "Database connection timeout due to network issues",
  "confidence": 0.85,
  "similar_defects": [
    {
      "score": 12.34,
      "content_preview": "ERROR: Database connection failed...",
      "line_range": "1-50",
      "metadata": {}
    }
  ],
  "reasoning": "The error pattern matches historical database timeout issues...",
  "recommendations": [
    "Check database server status",
    "Verify network connectivity",
    "Review connection pool settings"
  ]
}
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest test_defect_triage.py -v

# Run specific test class
pytest test_defect_triage.py::TestLogIngestor -v

# Run with coverage
pytest test_defect_triage.py --cov=. --cov-report=html
```

## Evaluation

Evaluate the agent's performance on a test dataset:

```bash
# Run evaluation with sample dataset
python evaluation.py

# This will generate:
# - evaluation_report.txt (detailed metrics)
# - evaluation_results.csv (per-case results)
# - evaluation_metrics.json (aggregated metrics)
```

### Custom Test Dataset

Create your own test dataset in JSON format:

```json
[
  {
    "id": "test_001",
    "error_log": "ERROR: Your error log here...",
    "ground_truth_root_cause": "Expected root cause",
    "ground_truth_category": "database",
    "severity": "high",
    "tags": ["database", "timeout"]
  }
]
```

Load and evaluate:

```python
from evaluation import load_test_cases_from_json, DefectTriageEvaluator

test_cases = load_test_cases_from_json("your_dataset.json")
results, metrics = await evaluator.evaluate_dataset(test_cases)
```

### Evaluation Metrics

The evaluation suite provides:

- **Accuracy**: Overall correctness of category predictions
- **Precision/Recall/F1**: Standard classification metrics
- **Confidence Calibration**: Correlation between confidence and correctness
- **Per-Category Performance**: Breakdown by error category
- **Response Time**: Average time per analysis

## Next Steps

1. ~~Implement `VectorMemory` class with FAISS~~ ✅
2. ~~Build Semantic Kernel agent with Azure OpenAI integration~~ ✅
3. ~~Add evaluation and testing suite~~ ✅
4. Build REST API for production deployment
5. Add web UI for interactive triage
6. Implement continuous learning from new defects

## Project Structure

```
AutonomousDefectTriageAgent/
├── log_ingestor.py          # Step 1: Log processing
├── vector_memory.py          # Step 2: Vector storage
├── defect_triage_agent.py   # Step 3: SK agent
├── test_defect_triage.py    # Unit tests
├── evaluation.py            # Evaluation suite
├── test_dataset.json        # Sample test cases
├── requirements.txt          # Dependencies
├── .env.example             # Configuration template
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Requirements

- Python 3.10+
- Azure OpenAI account with deployed model
- FAISS (CPU or GPU version)
- Semantic Kernel 1.0+
- Sentence Transformers

## License

MIT
