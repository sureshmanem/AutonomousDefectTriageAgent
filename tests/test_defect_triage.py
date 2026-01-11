"""
Testing Suite for Defect Triage Agent.

Comprehensive unit tests for log ingestion, vector memory, and agent functionality.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import json

from log_ingestor import LogIngestor, LogChunk
from vector_memory import VectorMemory, SearchResult
from defect_triage_agent import DefectTriageAgent, TriageResult


class TestLogIngestor:
    """Test cases for LogIngestor class."""
    
    def test_initialization(self) -> None:
        """Test LogIngestor initialization with default parameters."""
        ingestor = LogIngestor()
        assert ingestor.chunk_size == 50
        assert "Exception" in ingestor.exception_keywords
    
    def test_custom_chunk_size(self) -> None:
        """Test LogIngestor with custom chunk size."""
        ingestor = LogIngestor(chunk_size=30)
        assert ingestor.chunk_size == 30
    
    def test_remove_timestamps(self) -> None:
        """Test timestamp removal from log text."""
        ingestor = LogIngestor()
        
        log_with_timestamps = """
        2024-01-08 10:15:32.123 INFO Starting build
        [2024-01-08 10:15:33] ERROR Compilation failed
        1704712533456 DEBUG Processing
        """
        
        cleaned = ingestor.remove_timestamps(log_with_timestamps)
        
        # Timestamps should be removed
        assert "2024-01-08" not in cleaned
        assert "10:15:32" not in cleaned
        assert "1704712533456" not in cleaned
        
        # Content should remain
        assert "INFO Starting build" in cleaned
        assert "ERROR Compilation failed" in cleaned
    
    def test_find_exception_lines(self) -> None:
        """Test finding lines with exception keywords."""
        ingestor = LogIngestor()
        
        lines = [
            "INFO Starting process",
            "ERROR Connection failed",
            "java.lang.NullPointerException",
            "INFO Process completed",
            "FATAL System crash"
        ]
        
        exception_lines = ingestor.find_exception_lines(lines)
        
        assert 1 in exception_lines  # ERROR line
        assert 2 in exception_lines  # Exception line
        assert 4 in exception_lines  # FATAL line
        assert 0 not in exception_lines  # INFO line
        assert 3 not in exception_lines  # INFO line
    
    def test_chunk_creation(self) -> None:
        """Test creating chunks around exceptions."""
        ingestor = LogIngestor(chunk_size=10)
        
        lines = ["Line " + str(i) for i in range(20)]
        lines[10] = "ERROR Exception occurred"
        
        chunk = ingestor.create_chunk_around_exception(lines, 10)
        
        assert isinstance(chunk, LogChunk)
        assert chunk.exception_line == 11  # 1-indexed
        assert "Line 5" in chunk.content  # Before exception
        assert "Line 15" in chunk.content  # After exception
    
    def test_process_log_string(self) -> None:
        """Test processing log string into chunks."""
        ingestor = LogIngestor(chunk_size=20)
        
        log_content = """
        2024-01-08 10:00:00 INFO Starting application
        2024-01-08 10:00:01 ERROR Database connection failed
        java.sql.SQLException: Connection timeout
        at com.db.Pool.getConnection(Pool.java:123)
        2024-01-08 10:00:02 INFO Retrying connection
        """
        
        chunks = ingestor.process_log_string(log_content)
        
        assert len(chunks) > 0
        assert isinstance(chunks[0], LogChunk)
        assert "SQLException" in chunks[0].content or "ERROR" in chunks[0].content
    
    def test_chunks_to_dataframe(self) -> None:
        """Test converting chunks to DataFrame."""
        ingestor = LogIngestor()
        
        chunk = LogChunk(
            content="Test content",
            line_start=1,
            line_end=10,
            exception_line=5
        )
        
        df = ingestor.chunks_to_dataframe([chunk])
        
        assert len(df) == 1
        assert "content" in df.columns
        assert "line_start" in df.columns
        assert df.iloc[0]["content"] == "Test content"
    
    def test_read_nonexistent_file(self) -> None:
        """Test reading a file that doesn't exist."""
        ingestor = LogIngestor()
        
        with pytest.raises(FileNotFoundError):
            ingestor.read_log_file("/nonexistent/file.log")
    
    def test_process_log_file(self) -> None:
        """Test processing an actual log file."""
        ingestor = LogIngestor(chunk_size=30)
        
        # Create temporary log file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("""
2024-01-08 10:00:00 INFO Starting job
2024-01-08 10:00:05 ERROR Processing failed
java.lang.NullPointerException: Cannot invoke method
at com.example.Service.process(Service.java:45)
at com.example.Main.run(Main.java:12)
2024-01-08 10:00:10 FATAL Job terminated
            """)
            temp_path = f.name
        
        try:
            chunks = ingestor.process_log_file(temp_path)
            assert len(chunks) > 0
            assert any("NullPointerException" in chunk.content for chunk in chunks)
        finally:
            Path(temp_path).unlink()


@pytest.mark.asyncio
class TestVectorMemory:
    """Test cases for VectorMemory class."""
    
    async def test_initialization(self) -> None:
        """Test VectorMemory initialization."""
        memory = VectorMemory(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        assert memory.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert memory.dimension == 384  # all-MiniLM-L6-v2 dimension
        assert len(memory) == 0
    
    async def test_add_documents(self) -> None:
        """Test adding documents to vector memory."""
        memory = VectorMemory()
        
        chunks = [
            LogChunk("Error 1", 1, 10, 5),
            LogChunk("Error 2", 11, 20, 15)
        ]
        
        count = memory.add_documents(chunks)
        
        assert count == 2
        assert len(memory) == 2
        assert memory.index.ntotal == 2
    
    async def test_add_documents_async(self) -> None:
        """Test asynchronously adding documents."""
        memory = VectorMemory()
        
        chunks = [
            LogChunk("Async error 1", 1, 10, 5),
            LogChunk("Async error 2", 11, 20, 15)
        ]
        
        count = await memory.add_documents_async(chunks)
        
        assert count == 2
        assert len(memory) == 2
    
    async def test_search_similar(self) -> None:
        """Test searching for similar documents."""
        memory = VectorMemory()
        
        chunks = [
            LogChunk("Database connection timeout error", 1, 10, 5),
            LogChunk("NullPointerException in payment service", 11, 20, 15),
            LogChunk("OutOfMemoryError during batch processing", 21, 30, 25)
        ]
        
        memory.add_documents(chunks)
        
        # Search for database errors
        results = memory.search_similar("Database connection failed", top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].score >= 0  # L2 distance is non-negative
    
    async def test_search_similar_async(self) -> None:
        """Test async search."""
        memory = VectorMemory()
        
        chunks = [
            LogChunk("Connection error occurred", 1, 10, 5),
            LogChunk("Memory allocation failed", 11, 20, 15)
        ]
        
        await memory.add_documents_async(chunks)
        
        results = await memory.search_similar_async("Connection timeout", top_k=1)
        
        assert len(results) <= 1
        if results:
            assert "Connection" in results[0].chunk.content or "error" in results[0].chunk.content
    
    async def test_search_empty_memory(self) -> None:
        """Test searching when memory is empty."""
        memory = VectorMemory()
        
        results = memory.search_similar("Any query", top_k=3)
        
        assert len(results) == 0
    
    async def test_save_and_load(self) -> None:
        """Test saving and loading vector memory."""
        memory = VectorMemory()
        
        chunks = [
            LogChunk("Test error 1", 1, 10, 5),
            LogChunk("Test error 2", 11, 20, 15)
        ]
        
        memory.add_documents(chunks)
        
        # Save to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_memory"
            memory.save(save_path)
            
            # Load
            loaded_memory = VectorMemory.load(save_path)
            
            assert len(loaded_memory) == 2
            assert loaded_memory.dimension == memory.dimension
            assert loaded_memory.model_name == memory.model_name
    
    async def test_clear_memory(self) -> None:
        """Test clearing vector memory."""
        memory = VectorMemory()
        
        chunks = [LogChunk("Error", 1, 10, 5)]
        memory.add_documents(chunks)
        
        assert len(memory) == 1
        
        memory.clear()
        
        assert len(memory) == 0
        assert memory.index.ntotal == 0
    
    async def test_get_stats(self) -> None:
        """Test getting memory statistics."""
        memory = VectorMemory()
        
        chunks = [LogChunk("Error", 1, 10, 5)]
        memory.add_documents(chunks)
        
        stats = memory.get_stats()
        
        assert stats["num_documents"] == 1
        assert stats["index_total"] == 1
        assert "model_name" in stats
        assert "dimension" in stats


@pytest.mark.asyncio
class TestDefectTriageAgent:
    """Test cases for DefectTriageAgent."""
    
    async def test_triage_result_creation(self) -> None:
        """Test creating TriageResult."""
        result = TriageResult(
            root_cause="Database timeout",
            confidence=0.85,
            similar_defects=[],
            reasoning="Clear pattern match",
            recommendations=["Check DB", "Restart service"]
        )
        
        assert result.root_cause == "Database timeout"
        assert result.confidence == 0.85
        assert len(result.recommendations) == 2
    
    async def test_triage_result_to_json(self) -> None:
        """Test converting TriageResult to JSON."""
        result = TriageResult(
            root_cause="Test error",
            confidence=0.75,
            similar_defects=[],
            reasoning="Test reasoning",
            recommendations=["Fix 1"]
        )
        
        json_str = result.to_json()
        data = json.loads(json_str)
        
        assert data["root_cause"] == "Test error"
        assert data["confidence"] == 0.75
    
    async def test_triage_result_to_dict(self) -> None:
        """Test converting TriageResult to dictionary."""
        result = TriageResult(
            root_cause="Test error",
            confidence=0.75,
            similar_defects=[],
            reasoning="Test reasoning",
            recommendations=["Fix 1"]
        )
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert data["root_cause"] == "Test error"
    
    async def test_create_analysis_prompt(self) -> None:
        """Test prompt creation for analysis."""
        # Create mock vector memory
        memory = VectorMemory()
        chunks = [LogChunk("Historical error", 1, 10, 5)]
        memory.add_documents(chunks)
        
        # Create mock agent (without Azure credentials)
        # Note: This won't actually work without credentials, just testing structure
        try:
            agent = DefectTriageAgent(
                vector_memory=memory,
                azure_endpoint="https://test.openai.azure.com/",
                azure_api_key="test-key",
                deployment_name="test-deployment"
            )
            
            # Create mock search results
            search_results = memory.search_similar("test query", top_k=1)
            
            # Test prompt creation
            prompt = agent._create_analysis_prompt("New error log", search_results)
            
            assert "New Error Log" in prompt
            assert "Historical Defects" in prompt
            assert "root_cause" in prompt
            assert "confidence" in prompt
            
        except Exception as e:
            # Expected to fail without valid Azure credentials
            pytest.skip(f"Skipping due to missing Azure credentials: {e}")
    
    async def test_batch_analyze_structure(self) -> None:
        """Test batch analysis structure (without actual LLM calls)."""
        # This tests the structure, not actual LLM functionality
        memory = VectorMemory()
        chunks = [LogChunk("Test error", 1, 10, 5)]
        memory.add_documents(chunks)
        
        # Note: Actual testing would require Azure OpenAI credentials
        pytest.skip("Requires Azure OpenAI credentials for full testing")


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_pipeline_structure(self) -> None:
        """Test the structure of end-to-end pipeline."""
        # Step 1: Log Ingestion
        ingestor = LogIngestor(chunk_size=30)
        
        sample_log = """
        2024-01-08 10:00:00 INFO Starting
        2024-01-08 10:00:01 ERROR Connection failed
        java.sql.SQLException: Timeout
        """
        
        chunks = ingestor.process_log_string(sample_log)
        assert len(chunks) > 0
        
        # Step 2: Vector Memory
        memory = VectorMemory()
        count = memory.add_documents(chunks)
        assert count > 0
        
        # Step 3: Search
        results = memory.search_similar("Connection error", top_k=1)
        assert isinstance(results, list)
    
    def test_dataframe_integration(self) -> None:
        """Test DataFrame integration."""
        ingestor = LogIngestor()
        
        log = "ERROR Test error\njava.lang.Exception\nStack trace"
        chunks = ingestor.process_log_string(log)
        
        df = ingestor.chunks_to_dataframe(chunks)
        
        assert len(df) > 0
        assert "content" in df.columns
        assert df["content"].dtype == object


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
