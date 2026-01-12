# Autonomous Defect Triage Agent - Cheat Sheet

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# Test the agent
python src/defect_triage_agent.py
```

## 📦 Core Components

| Component | File | Purpose |
|-----------|------|------|
| **Log Ingestor** | `src/log_ingestor.py` | Parse & chunk Jenkins logs |
| **Vector Memory** | `src/vector_memory.py` | FAISS-based similarity search |
| **Triage Agent** | `src/defect_triage_agent.py` | Semantic Kernel + Azure OpenAI |

## 🔧 Environment Variables

```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-01  # Optional
```

## 📝 Log Ingestor

### Basic Usage
```python
from src.log_ingestor import LogIngestor

ingestor = LogIngestor(chunk_size=50)

# From file
chunks = ingestor.process_log_file("jenkins.log")

# From string
chunks = ingestor.process_log_string(log_content)

# To DataFrame
df = ingestor.chunks_to_dataframe(chunks)
```

### Key Methods
| Method | Parameters | Returns |
|--------|-----------|---------|
| `process_log_file()` | `file_path: str` | `List[LogChunk]` |
| `process_log_string()` | `log_content: str` | `List[LogChunk]` |
| `chunks_to_dataframe()` | `chunks: List[LogChunk]` | `pd.DataFrame` |

## 🗄️ Vector Memory

### Initialization
```python
from src.vector_memory import VectorMemory

memory = VectorMemory(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    index_type="flat"  # Options: flat, ivf, hnsw
)
```

### Index Types
| Type | Best For | Speed | Accuracy |
|------|----------|-------|----------|
| `flat` | Small datasets (<10K) | Fast | 100% |
| `ivf` | Medium datasets (10K-1M) | Medium | ~95% |
| `hnsw` | Large datasets (>1M) | Very Fast | ~98% |

### Operations
```python
# Add documents (async)
await memory.add_documents_async(chunks)

# Search (async)
results = await memory.search_similar_async(query, top_k=3)

# Save/Load
memory.save("./vector_db")
loaded = VectorMemory.load("./vector_db")
```

## 🤖 Defect Triage Agent

### Initialization
```python
from src.defect_triage_agent import DefectTriageAgent

agent = DefectTriageAgent(
    vector_memory=memory,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.3,      # Lower = more deterministic
    max_tokens=1500
)
```

### Analyze New Error (Async)
```python
result = await agent.analyze_defect_async(
    new_error_log="NullPointerException...",
    top_k=3  # Number of similar defects to retrieve
)

print(result.root_cause)
print(f"Confidence: {result.confidence}")
print(result.recommendations)
```

### Analyze New Error (Sync)
```python
result = agent.analyze_defect(
    new_error_log="NullPointerException...",
    top_k=3
)
```

### TriageResult Object
```python
@dataclass
class TriageResult:
    root_cause: str              # Root cause description
    confidence: float            # 0.0 to 1.0
    similar_defects: List[Dict]  # Retrieved historical defects
    reasoning: str               # Analysis explanation
    recommendations: List[str]   # Actionable steps

# Convert to JSON/Dict
result.to_json()  # JSON string
result.to_dict()  # Python dict
```

## 🧪 Testing & Evaluation

### Run Tests
```bash
pytest tests/
pytest tests/test_defect_triage.py -v
```

### Run Evaluation
```bash
python evaluation/evaluation.py
```

### Evaluation Metrics
- **Accuracy**: Correct root cause identification
- **Precision**: Relevant similar defects retrieved
- **Recall**: Coverage of true matches
- **F1-Score**: Harmonic mean of precision/recall
- **Confidence Correlation**: Alignment with ground truth

## 📊 Data Structure

### LogChunk
```python
@dataclass
class LogChunk:
    content: str
    line_start: int
    line_end: int
    metadata: Dict[str, Any]
```

### SearchResult
```python
@dataclass
class SearchResult:
    chunk: LogChunk
    score: float  # Similarity score (higher = more similar)
```

## 🐳 Docker Deployment

```bash
# Build
docker-compose build

# Run
docker-compose up -d

# Stop
docker-compose down
```

**Ports:**
- API Server: `8000`
- Vector DB Volume: `./data/vector_db`

## 🔍 Common Patterns

### Complete Workflow
```python
import asyncio
from src.log_ingestor import LogIngestor
from src.vector_memory import VectorMemory
from src.defect_triage_agent import DefectTriageAgent

async def main():
    # 1. Ingest historical logs
    ingestor = LogIngestor()
    chunks = ingestor.process_log_file("historical_logs.log")
    
    # 2. Build vector memory
    memory = VectorMemory()
    await memory.add_documents_async(chunks)
    memory.save("./vector_db")
    
    # 3. Initialize agent
    agent = DefectTriageAgent(
        vector_memory=memory,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    )
    
    # 4. Analyze new error
    new_error = "NullPointerException at line 42..."
    result = await agent.analyze_defect_async(new_error)
    
    print(result.to_json())

asyncio.run(main())
```

### Batch Processing
```python
# Process multiple log files
log_files = ["build1.log", "build2.log", "build3.log"]
all_chunks = []

for log_file in log_files:
    chunks = ingestor.process_log_file(log_file)
    all_chunks.extend(chunks)

await memory.add_documents_async(all_chunks)
```

## ⚡ Performance Tips

1. **Use Async Operations**: Always prefer `_async()` methods for non-blocking I/O
2. **Choose Right Index Type**: 
   - `flat` for <10K documents
   - `ivf` for 10K-1M documents
   - `hnsw` for >1M documents
3. **Batch Operations**: Add documents in batches instead of one-by-one
4. **Adjust Temperature**: 
   - Lower (0.1-0.3) for deterministic analysis
   - Higher (0.5-0.7) for creative solutions
5. **Cache Vector DB**: Save/load to avoid rebuilding

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Azure OpenAI connection fails | Check endpoint URL, API key, and deployment name |
| Vector DB not found | Run `memory.save()` before `load()` |
| Low confidence scores | Add more historical defects to improve RAG |
| Slow searches | Switch to IVF or HNSW index for large datasets |
| Import errors | Ensure all dependencies: `pip install -r requirements.txt` |
| Memory issues | Process logs in smaller batches |

## 📚 Key Dependencies

```txt
semantic-kernel==1.3.0
azure-ai-openai>=1.0.0
faiss-cpu==1.8.0
sentence-transformers==2.2.2
pandas==2.1.3
numpy==1.26.2
```

## 🎯 Best Practices

1. **Preprocessing**: Always remove timestamps/noise from logs
2. **Chunking**: Use 50-line chunks centered on errors
3. **Top-K**: Retrieve 3-5 similar defects for best context
4. **Confidence Threshold**: Only act on results with >0.7 confidence
5. **Regular Updates**: Continuously add new resolved defects to vector DB
6. **Error Handling**: Wrap agent calls in try-except for production
7. **Logging**: Enable detailed logging for debugging
8. **Monitoring**: Track confidence scores and retrieval quality

## 🔗 Quick Links

- **Documentation**: `/docs/`
- **Technical Interview**: `/docs/TechnicalInterview.md`
- **Mock Interview**: `/docs/MockInterview.md`
- **Test Dataset**: `/tests/test_dataset.json`
- **Evaluation Results**: `/evaluation/evaluation_results.csv`

---

**Version**: 1.0.0  
**Framework**: Microsoft Semantic Kernel  
**LLM**: Azure OpenAI (GPT-4)  
**Vector DB**: FAISS  
**Pattern**: RAG (Retrieval-Augmented Generation)
