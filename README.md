# Autonomous Defect Triage Agent

An Agentic AI system that automates the analysis of Jenkins failure logs using Microsoft Semantic Kernel and Azure OpenAI.

## Architecture

### 1. Log Ingestor (`src/log_ingestor.py`) ✅
- Reads Jenkins failure logs
- Removes timestamps using regex patterns
- Chunks text into 50-line blocks centered around Exception keywords
- Uses strict type hinting and Python 3.10+ syntax

### 2. Vector Memory (`src/vector_memory.py`) ✅
- FAISS-based vector storage with multiple index types
- Sentence-transformers embeddings (all-MiniLM-L6-v2)
- Methods: `add_documents()`, `search_similar()`, `save()`, `load()`
- Full async/await support for non-blocking operations

### 3. Semantic Kernel Agent (`src/defect_triage_agent.py`) ✅
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
from src.log_ingestor import LogIngestor

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
from src.vector_memory import VectorMemory
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
from src.defect_triage_agent import DefectTriageAgent
from src.vector_memory import VectorMemory
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

### Step 4: Run REST API Server

```bash
# Start the API server
python src/api.py

# Or use uvicorn directly
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Access the API documentation
# Open http://localhost:8000/docs in your browser
```

#### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /stats` - System statistics
- `POST /analyze` - Analyze single error log
- `POST /analyze/batch` - Batch analysis
- `POST /add-defect` - Add defect to knowledge base
- `POST /clear-knowledge-base` - Clear knowledge base

#### Example API Usage

```python
from src.api_client import DefectTriageClient

client = DefectTriageClient("http://localhost:8000")

# Analyze a defect
result = client.analyze_defect(
    error_log="ERROR: Database connection timeout...",
    top_k=3
)

print(f"Root Cause: {result['root_cause']}")
print(f"Confidence: {result['confidence']:.2%}")
```

#### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Analyze defect
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "error_log": "ERROR: Database connection failed...",
    "top_k": 3,
    "include_similar": true
  }'

# Add new defect
curl -X POST http://localhost:8000/add-defect \
  -H "Content-Type: application/json" \
  -d '{
    "error_log": "ERROR: New error...",
    "source": "Jenkins",
    "metadata": {"severity": "high"}
  }'
```

## Quick Test

Run the complete pipeline:

```bash
# Test log ingestion
python src/log_ingestor.py

# Test vector memory
python src/vector_memory.py

# Test defect triage agent (requires Azure OpenAI)
python src/defect_triage_agent.py
```

## Complete Example

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# 2. Run end-to-end triage
python src/defect_triage_agent.py
```

## Docker Deployment

### Build and Run with Docker

```bash
# Build the Docker image
docker build -f deployment/Dockerfile -t defect-triage-api .

# Run the container
docker run -d \
  --name defect-triage \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data/vector_db:/app/data/vector_db \
  -v $(pwd)/data/logs:/app/data/logs \
  defect-triage-api

# View logs
docker logs -f defect-triage

# Stop the container
docker stop defect-triage
```

### Using Docker Compose

```bash
# Navigate to deployment directory
cd deployment

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# Return to project root
cd ..
```

### Production Deployment

For production deployment:

1. **Use a production WSGI server** (already using Uvicorn)
2. **Set up reverse proxy** (Nginx/Traefik)
3. **Enable HTTPS** with SSL certificates
4. **Configure CORS** appropriately in `api.py`
5. **Set up monitoring** (health checks, metrics)
6. **Use persistent volumes** for vector database
7. **Implement rate limiting**
8. **Add authentication** (API keys, OAuth)

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
- ✅ Production-ready REST API with FastAPI
- ✅ Docker containerization support
- ✅ Auto-generated API documentation (OpenAPI/Swagger)
- ✅ Health checks and monitoring endpoints
- ✅ Batch processing capabilities
- ✅ Background task processing

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
pytest tests/test_defect_triage.py -v

# Run specific test class
pytest tests/test_defect_triage.py::TestLogIngestor -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Evaluation

Evaluate the agent's performance on a test dataset:

```bash
# Run evaluation with sample dataset
python evaluation/evaluation.py

# This will generate:
# - evaluation/evaluation_report.txt (detailed metrics)
# - evaluation/evaluation_results.csv (per-case results)
# - evaluation/evaluation_metrics.json (aggregated metrics)
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
from evaluation.evaluation import load_test_cases_from_json, DefectTriageEvaluator

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
4. ~~Build REST API for production deployment~~ ✅
5. Add web UI for interactive triage
6. Implement continuous learning from new defects
7. Add Kubernetes deployment manifests
8. Implement request caching and rate limiting

## Project Structure

```
AutonomousDefectTriageAgent/
├── src/                      # Source code
│   ├── __init__.py
│   ├── log_ingestor.py       # Step 1: Log processing
│   ├── vector_memory.py      # Step 2: Vector storage
│   ├── defect_triage_agent.py # Step 3: SK agent
│   ├── api.py                # REST API server
│   └── api_client.py         # API client example
├── tests/                    # Test files
│   ├── __init__.py
│   ├── test_defect_triage.py # Unit tests
│   └── test_dataset.json     # Sample test cases
├── evaluation/               # Evaluation suite
│   ├── __init__.py
│   ├── evaluation.py         # Evaluation script
│   ├── evaluation_metrics.json
│   ├── evaluation_report.txt
│   └── evaluation_results.csv
├── deployment/               # Docker & deployment
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/                     # Documentation
│   └── MockInterview.md
├── data/                     # Data storage
│   ├── logs/                 # Log files
│   └── vector_db/            # FAISS index
├── requirements.txt          # Dependencies
├── .env.example              # Configuration template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Requirements

- Python 3.10+
- Azure OpenAI account with deployed model
- FAISS (CPU or GPU version)
- Semantic Kernel 1.0+
- Sentence Transformers

## License

MIT
