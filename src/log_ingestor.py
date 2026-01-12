"""
Log Ingestor Module for Defect Triage System.

This module provides functionality to ingest Jenkins failure logs,
clean them by removing timestamps, and chunk them around Exception keywords.
"""

import re
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class LogChunk:
    """Represents a chunk of log data centered around an Exception."""
    
    content: str
    line_start: int
    line_end: int
    exception_line: int
    
    def __str__(self) -> str:
        return (
            f"LogChunk(lines {self.line_start}-{self.line_end}, "
            f"exception at line {self.exception_line})"
        )


class LogIngestor:
    """
    Ingests and processes Jenkins failure logs for defect triage.
    
    This class reads log files, removes timestamps, and creates chunks
    of text centered around Exception keywords for further analysis.
    """
    
    def __init__(
        self, 
        chunk_size: int = 50,
        exception_keywords: Optional[List[str]] = None
    ) -> None:
        """
        Initialize the LogIngestor.
        
        Args:
            chunk_size: Number of lines per chunk (centered around exception)
            exception_keywords: List of keywords to identify exceptions
        """
        logger.info(f"Initializing LogIngestor with chunk_size={chunk_size}")
        self.chunk_size = chunk_size
        self.exception_keywords = exception_keywords or [
            "Exception", "Error", "ERROR", "FATAL", "FAILED"
        ]
        
        # Comprehensive timestamp patterns
        self.timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}[.,]?\d*',  # ISO format
            r'\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}',             # MM/DD/YYYY HH:MM:SS
            r'\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]',         # [YYYY-MM-DD HH:MM:SS]
            r'\d{2}:\d{2}:\d{2}[.,]\d+',                           # HH:MM:SS.mmm
            r'\d{13,}',                                             # Unix timestamp (ms)
        ]
        self.timestamp_regex = re.compile('|'.join(self.timestamp_patterns))
    
    def read_log_file(self, file_path: str | Path) -> str:
        """
        Read the log file content.
        
        Args:
            file_path: Path to the log file
            
        Returns:
            Raw content of the log file
            
        Raises:
            FileNotFoundError: If the log file doesn't exist
            IOError: If there's an error reading the file
        """
        logger.debug(f"Reading log file: {file_path}")
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"Log file not found: {file_path}")
            raise FileNotFoundError(f"Log file not found: {file_path}")
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                logger.info(f"Successfully read log file: {file_path}, size: {len(content)} bytes")
                return content
        except Exception as e:
            logger.error(f"Error reading log file {file_path}: {e}")
            raise IOError(f"Error reading log file {file_path}: {e}")
    
    def remove_timestamps(self, text: str) -> str:
        """
        Remove timestamps from log text using regex patterns.
        
        Args:
            text: Raw log text with timestamps
            
        Returns:
            Cleaned text with timestamps removed
        """
        logger.debug(f"Removing timestamps from text of length {len(text)}")
        cleaned_text = self.timestamp_regex.sub('', text)
        # Clean up multiple spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        # Clean up spaces at line starts
        cleaned_text = re.sub(r'^\s+', '', cleaned_text, flags=re.MULTILINE)
        return cleaned_text
    
    def find_exception_lines(self, lines: List[str]) -> List[int]:
        """
        Find line numbers that contain exception keywords.
        
        Args:
            lines: List of log lines
            
        Returns:
            List of line numbers (0-indexed) containing exceptions
        """
        logger.debug(f"Finding exception lines in {len(lines)} lines")
        exception_lines: List[int] = []
        
        for idx, line in enumerate(lines):
            if any(keyword in line for keyword in self.exception_keywords):
                exception_lines.append(idx)
        
        logger.info(f"Found {len(exception_lines)} exception lines")
        return exception_lines
    
    def create_chunk_around_exception(
        self, 
        lines: List[str], 
        exception_line: int
    ) -> LogChunk:
        """
        Create a chunk of lines centered around an exception.
        
        Args:
            lines: All log lines
            exception_line: Line number of the exception (0-indexed)
            
        Returns:
            LogChunk object containing the chunk and metadata
        """
        logger.debug(f"Creating chunk around exception at line {exception_line}")
        total_lines = len(lines)
        half_chunk = self.chunk_size // 2
        
        # Calculate start and end indices
        start_idx = max(0, exception_line - half_chunk)
        end_idx = min(total_lines, exception_line + half_chunk)
        
        # Extract chunk
        chunk_lines = lines[start_idx:end_idx]
        chunk_content = '\n'.join(chunk_lines)
        
        return LogChunk(
            content=chunk_content,
            line_start=start_idx + 1,  # 1-indexed for display
            line_end=end_idx,
            exception_line=exception_line + 1
        )
    
    def chunk_log_text(self, text: str) -> List[LogChunk]:
        """
        Split log text into chunks centered around exceptions.
        
        Args:
            text: Cleaned log text
            
        Returns:
            List of LogChunk objects
        """
        logger.debug(f"Chunking log text of length {len(text)}")
        lines = text.split('\n')
        exception_lines = self.find_exception_lines(lines)
        
        if not exception_lines:
            # If no exceptions found, return entire log as single chunk
            return [LogChunk(
                content=text,
                line_start=1,
                line_end=len(lines),
                exception_line=-1
            )]
        
        chunks: List[LogChunk] = []
        seen_ranges = set()
        
        for exc_line in exception_lines:
            chunk = self.create_chunk_around_exception(lines, exc_line)
            
            # Avoid duplicate chunks with overlapping ranges
            range_key = (chunk.line_start, chunk.line_end)
            if range_key not in seen_ranges:
                chunks.append(chunk)
                seen_ranges.add(range_key)
        
        logger.info(f"Created {len(chunks)} chunks from log text")
        return chunks
    
    def process_log_file(self, file_path: str | Path) -> List[LogChunk]:
        """
        Complete pipeline: read, clean, and chunk a log file.
        
        Args:
            file_path: Path to the log file
            
        Returns:
            List of LogChunk objects ready for vector embedding
        """
        logger.info(f"Processing log file: {file_path}")
        # Read file
        raw_content = self.read_log_file(file_path)
        
        # Remove timestamps
        cleaned_content = self.remove_timestamps(raw_content)
        
        # Create chunks
        chunks = self.chunk_log_text(cleaned_content)
        
        return chunks
    
    def process_log_string(self, log_content: str) -> List[LogChunk]:
        """
        Process log content from a string (useful for testing or API input).
        
        Args:
            log_content: Raw log content as string
            
        Returns:
            List of LogChunk objects
        """
        logger.info(f"Processing log string of length {len(log_content)}")
        cleaned_content = self.remove_timestamps(log_content)
        chunks = self.chunk_log_text(cleaned_content)
        return chunks
    
    def chunks_to_dataframe(self, chunks: List[LogChunk]) -> pd.DataFrame:
        """
        Convert LogChunk objects to a Pandas DataFrame.
        
        Args:
            chunks: List of LogChunk objects
            
        Returns:
            DataFrame with columns: content, line_start, line_end, exception_line
        """
        logger.debug(f"Converting {len(chunks)} chunks to DataFrame")
        data = {
            'content': [chunk.content for chunk in chunks],
            'line_start': [chunk.line_start for chunk in chunks],
            'line_end': [chunk.line_end for chunk in chunks],
            'exception_line': [chunk.exception_line for chunk in chunks],
        }
        
        return pd.DataFrame(data)


# Example usage
if __name__ == "__main__":
    # Create ingestor instance
    ingestor = LogIngestor(chunk_size=50)
    
    # Example log content
    sample_log = """
2024-01-08 10:15:32.123 INFO Starting Jenkins build
2024-01-08 10:15:33.456 INFO Compiling source code
2024-01-08 10:15:45.789 ERROR Compilation failed
java.lang.NullPointerException: Cannot invoke method on null object
    at com.example.Service.process(Service.java:45)
    at com.example.Controller.handle(Controller.java:23)
2024-01-08 10:15:46.012 FATAL Build terminated
    """
    
    # Process log
    chunks = ingestor.process_log_string(sample_log)
    
    # Display results
    print(f"Found {len(chunks)} chunk(s):\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}")
        print(f"Content preview: {chunk.content[:100]}...")
        print("-" * 80)
    
    # Convert to DataFrame
    df = ingestor.chunks_to_dataframe(chunks)
    print("\nDataFrame:")
    print(df.head())
