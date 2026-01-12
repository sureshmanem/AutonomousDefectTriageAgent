# Logging Documentation

## Overview

The Autonomous Defect Triage Agent now includes comprehensive logging throughout all components for debugging, monitoring, and understanding the system's behavior.

## Logging Architecture

### Components with Logging

1. **log_ingestor.py** - Logs file reading, timestamp removal, chunking operations
2. **vector_memory.py** - Logs embedding generation, FAISS operations, search queries
3. **defect_triage_agent.py** - Logs analysis requests, LLM interactions, results
4. **evaluation.py** - Logs evaluation progress, metrics calculation

### Log Levels

The system uses standard Python logging levels:

| Level | Usage | Example |
|-------|-------|---------|
| **DEBUG** | Detailed diagnostic information | Parameter values, internal state |
| **INFO** | General informational messages | Operation started/completed, counts |
| **WARNING** | Unexpected but handled situations | Missing optional data, fallbacks |
| **ERROR** | Errors that affect functionality | Failed operations, exceptions |
| **CRITICAL** | Severe errors requiring attention | System failures, data corruption |

## Configuration

### Basic Setup

```python
from logging_config import setup_logging

# Default configuration (INFO level, logs to file and console)
setup_logging()

# Custom configuration
setup_logging(
    log_level="DEBUG",
    log_to_file=True,
    log_dir="./data/logs",
    log_format="detailed"
)
```

### Environment Variables

You can also configure logging via environment variables:

```bash
export LOG_LEVEL=DEBUG
export LOG_TO_FILE=true
export LOG_DIR=./data/logs
export LOG_FORMAT=detailed
```

### Format Options

1. **simple**: `LEVEL - LOGGER - MESSAGE`
2. **detailed**: `TIMESTAMP - LOGGER - LEVEL - [FILE:LINE] - MESSAGE`
3. **json**: Structured JSON format for log aggregation tools

## Usage Patterns

### In Source Code

Every major function now includes logging:

```python
import logging
logger = logging.getLogger(__name__)

def process_log_file(self, file_path: str) -> List[LogChunk]:
    logger.info(f"Processing log file: {file_path}")
    
    try:
        chunks = self._do_processing(file_path)
        logger.info(f"Successfully processed {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        raise
```

### Function Entry Points

All functions log:
- **Entry**: Function name and key parameters
- **Progress**: Important intermediate steps
- **Exit**: Results, counts, or success indicators
- **Errors**: Exceptions with context

### Examples by Component

#### 1. Log Ingestor
```
INFO - log_ingestor - Initializing LogIngestor with chunk_size=50
DEBUG - log_ingestor - Reading log file: /path/to/file.log
INFO - log_ingestor - Successfully read log file: file.log, size: 15234 bytes
DEBUG - log_ingestor - Removing timestamps from text of length 15234
DEBUG - log_ingestor - Finding exception lines in 342 lines
INFO - log_ingestor - Found 5 exception lines
INFO - log_ingestor - Created 5 chunks from log text
```

#### 2. Vector Memory
```
INFO - vector_memory - Initializing VectorMemory with model=all-MiniLM-L6-v2, index_type=flat
DEBUG - vector_memory - Creating FAISS index of type: flat
DEBUG - vector_memory - Encoding 10 texts into embeddings
INFO - vector_memory - Adding 10 documents to vector database
DEBUG - vector_memory - Searching for similar documents with query length 256, top_k=3
INFO - vector_memory - Found 3 similar documents
INFO - vector_memory - Saving vector memory to ./vector_db
```

#### 3. Defect Triage Agent
```
INFO - defect_triage_agent - Initializing DefectTriageAgent with deployment=gpt-4, temperature=0.3
INFO - defect_triage_agent - Analyzing defect with top_k=3, error_log_length=1024
DEBUG - defect_triage_agent - Creating analysis prompt with 3 similar defects
INFO - defect_triage_agent - Batch analyzing 5 error logs
INFO - defect_triage_agent - Generating summary report for 5 results
```

#### 4. Evaluation
```
INFO - evaluation - Initializing DefectTriageEvaluator
INFO - evaluation - Evaluating agent on 25 test cases
DEBUG - evaluation - Evaluating case: test_001
INFO - evaluation - Generating evaluation report for 25 results
INFO - evaluation - Exporting 25 results to CSV: results.csv
```

## Log Files

### Location
Logs are saved to `./data/logs/` by default with rotation:
- `defect_triage_YYYYMMDD.log` - Current day's log
- `defect_triage_YYYYMMDD.log.1` - Previous rotation
- `defect_triage_YYYYMMDD.log.2` - Older rotation
- etc. (up to 5 backup files)

### Rotation
- **Max Size**: 10 MB per file
- **Backup Count**: 5 files
- **Automatic**: Rotates when size limit is reached

## Viewing Logs

### Real-time Monitoring
```bash
# Follow log file
tail -f data/logs/defect_triage_$(date +%Y%m%d).log

# Filter by level
tail -f data/logs/*.log | grep "ERROR"

# Filter by component
tail -f data/logs/*.log | grep "vector_memory"
```

### Search Logs
```bash
# Find all errors
grep "ERROR" data/logs/*.log

# Find specific operation
grep "Analyzing defect" data/logs/*.log

# Count log entries by level
grep -o "INFO\|WARNING\|ERROR\|DEBUG" data/logs/*.log | sort | uniq -c
```

### Analysis
```bash
# Show only timestamps and messages
awk -F' - ' '{print $1, $NF}' data/logs/*.log

# Extract errors with context
grep -A 3 -B 3 "ERROR" data/logs/*.log
```

## Performance Logging

The system logs timing information for performance monitoring:

```python
import time

logger.info("Starting batch processing")
start_time = time.time()

# ... operation ...

duration = time.time() - start_time
logger.info(f"Batch processing completed in {duration:.2f}s")
```

## Debugging Tips

### 1. Enable DEBUG Level
```python
setup_logging(log_level="DEBUG")
```

### 2. Component-Specific Debugging
```python
import logging

# Enable debug for specific module
logging.getLogger("vector_memory").setLevel(logging.DEBUG)

# Disable noisy library
logging.getLogger("urllib3").setLevel(logging.WARNING)
```

### 3. Temporary Debug Logging
```python
logger.debug("Variable state: %s", expensive_debug_info())

# Better: Only compute if needed
if logger.isEnabledFor(logging.DEBUG):
    debug_info = expensive_operation()
    logger.debug(f"Debug info: {debug_info}")
```

### 4. Exception Tracing
```python
try:
    risky_operation()
except Exception:
    logger.exception("Operation failed")  # Includes full traceback
```

## Integration with Tools

### Log Aggregation (ELK Stack, Splunk)
Use JSON format for easy parsing:
```python
setup_logging(log_format="json")
```

### Docker Logs
```bash
docker logs -f container_name
docker logs --tail 100 container_name | grep "ERROR"
```

### CI/CD Pipelines
Logs are automatically captured by most CI/CD systems. Ensure:
- Use appropriate log levels (INFO for status, ERROR for failures)
- Include context in error messages
- Log performance metrics for tracking

## Best Practices

1. **Use Appropriate Levels**
   - DEBUG: Development/troubleshooting only
   - INFO: Important business events
   - WARNING: Recoverable issues
   - ERROR: Failed operations
   - CRITICAL: System-level failures

2. **Include Context**
   ```python
   # Good
   logger.error(f"Failed to process file {filename}: {error}")
   
   # Bad
   logger.error("Error occurred")
   ```

3. **Avoid Logging Sensitive Data**
   ```python
   # Bad
   logger.info(f"User password: {password}")
   
   # Good
   logger.info(f"User authenticated: {username}")
   ```

4. **Use Structured Logging**
   ```python
   logger.info("Request completed", extra={
       "user_id": user_id,
       "duration_ms": duration,
       "status": "success"
   })
   ```

5. **Performance Considerations**
   - Use f-strings for formatting (lazy evaluation)
   - Avoid logging in tight loops at INFO level
   - Use DEBUG for verbose output

## Troubleshooting Common Issues

### No Logs Appearing
- Check log level configuration
- Verify log directory permissions
- Ensure `setup_logging()` is called before operations

### Too Many Logs
- Increase log level to WARNING or ERROR
- Reduce third-party library verbosity
- Use log rotation settings

### Missing Context
- Add more logger.info() statements at key points
- Include relevant variables in log messages
- Use logger.exception() for errors

## Example: Full Application Setup

```python
#!/usr/bin/env python3
import os
from logging_config import setup_logging
from defect_triage_agent import DefectTriageAgent
from vector_memory import VectorMemory

# Configure logging at application startup
log_level = os.getenv("LOG_LEVEL", "INFO")
setup_logging(
    log_level=log_level,
    log_to_file=True,
    log_dir="./data/logs",
    log_format="detailed"
)

# Now all components will log appropriately
memory = VectorMemory()
agent = DefectTriageAgent(memory, ...)
result = agent.analyze_defect(error_log)
```

## Monitoring Production Systems

For production deployments:

1. **Set to INFO or WARNING** to reduce log volume
2. **Enable file logging** with rotation
3. **Monitor ERROR and CRITICAL** logs actively
4. **Track performance metrics** from log timestamps
5. **Archive logs** for compliance and analysis
6. **Use log aggregation tools** for centralized monitoring

## Summary

The logging system provides:
- ✅ Complete visibility into system operations
- ✅ Debugging support at all levels
- ✅ Performance monitoring capabilities
- ✅ Error tracking and diagnostics
- ✅ Audit trail for operations
- ✅ Production-ready log management

All components now include strategic logging at function entry/exit points, error conditions, and significant operations.
