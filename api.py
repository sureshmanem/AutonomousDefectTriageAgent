"""
REST API for Defect Triage Agent.

FastAPI-based REST API for production deployment of the autonomous defect triage system.
"""

import asyncio
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from defect_triage_agent import DefectTriageAgent, TriageResult
from vector_memory import VectorMemory
from log_ingestor import LogIngestor, LogChunk
from evaluation import DefectTriageEvaluator, EvaluationCase


# Pydantic models for API requests/responses
class AnalyzeRequest(BaseModel):
    """Request model for analyzing an error log."""
    
    error_log: str = Field(..., min_length=10, description="The error log to analyze")
    top_k: int = Field(3, ge=1, le=10, description="Number of similar defects to retrieve")
    include_similar: bool = Field(True, description="Include similar defects in response")
    
    @validator('error_log')
    def validate_error_log(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("Error log must be at least 10 characters")
        return v


class TriageResponse(BaseModel):
    """Response model for triage analysis."""
    
    root_cause: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    recommendations: List[str]
    similar_defects: Optional[List[Dict[str, Any]]] = None
    analysis_time_ms: float
    timestamp: str


class BatchAnalyzeRequest(BaseModel):
    """Request model for batch analysis."""
    
    error_logs: List[str] = Field(..., min_items=1, max_items=50)
    top_k: int = Field(3, ge=1, le=10)
    
    @validator('error_logs')
    def validate_error_logs(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one error log is required")
        if len(v) > 50:
            raise ValueError("Maximum 50 error logs per batch")
        return v


class BatchTriageResponse(BaseModel):
    """Response model for batch analysis."""
    
    results: List[TriageResponse]
    total_count: int
    total_time_ms: float
    timestamp: str


class AddDefectRequest(BaseModel):
    """Request model for adding a defect to knowledge base."""
    
    error_log: str = Field(..., min_length=10)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    source: Optional[str] = Field(None, description="Source system (e.g., Jenkins, GitHub)")


class AddDefectResponse(BaseModel):
    """Response model for adding defects."""
    
    chunks_added: int
    total_defects: int
    message: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    vector_db_size: int
    model_name: str
    uptime_seconds: float
    timestamp: str


class StatsResponse(BaseModel):
    """Statistics response."""
    
    total_defects: int
    model_name: str
    index_type: str
    embedding_dimension: int
    api_version: str


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str
    detail: Optional[str] = None
    timestamp: str


# Global state
class AppState:
    """Application state management."""
    
    def __init__(self):
        self.agent: Optional[DefectTriageAgent] = None
        self.vector_memory: Optional[VectorMemory] = None
        self.log_ingestor: Optional[LogIngestor] = None
        self.start_time: float = time.time()
        self.is_ready: bool = False


app_state = AppState()


# FastAPI app
app = FastAPI(
    title="Autonomous Defect Triage API",
    description="REST API for automated defect triage using AI and vector similarity search",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency for getting agent
async def get_agent() -> DefectTriageAgent:
    """Dependency to get the initialized agent."""
    if not app_state.is_ready or app_state.agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready. Agent not initialized."
        )
    return app_state.agent


async def get_vector_memory() -> VectorMemory:
    """Dependency to get vector memory."""
    if not app_state.is_ready or app_state.vector_memory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready. Vector memory not initialized."
        )
    return app_state.vector_memory


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    import os
    from dotenv import load_dotenv
    
    print("🚀 Starting Defect Triage API...")
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    vector_db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")
    
    # Validate configuration
    if not all([azure_endpoint, azure_api_key, deployment_name]):
        print("❌ Error: Missing Azure OpenAI credentials")
        print("Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME")
        return
    
    try:
        # Load or create vector memory
        vector_db_path = Path(vector_db_path)
        
        if vector_db_path.exists():
            print(f"📚 Loading vector database from {vector_db_path}...")
            app_state.vector_memory = VectorMemory.load(vector_db_path)
        else:
            print("📚 Creating new vector database...")
            app_state.vector_memory = VectorMemory()
            print("⚠️  Warning: Starting with empty knowledge base. Add defects via /add-defect endpoint.")
        
        # Initialize agent
        print("🤖 Initializing Defect Triage Agent...")
        app_state.agent = DefectTriageAgent(
            vector_memory=app_state.vector_memory,
            azure_endpoint=azure_endpoint,
            azure_api_key=azure_api_key,
            deployment_name=deployment_name,
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "1500"))
        )
        
        # Initialize log ingestor
        app_state.log_ingestor = LogIngestor()
        
        app_state.is_ready = True
        print("✅ Service ready!")
        print(f"📊 Vector DB size: {len(app_state.vector_memory)} defects")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        app_state.is_ready = False


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("🛑 Shutting down Defect Triage API...")
    
    # Save vector memory if modified
    if app_state.vector_memory and len(app_state.vector_memory) > 0:
        try:
            import os
            vector_db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")
            app_state.vector_memory.save(vector_db_path)
            print(f"💾 Vector database saved to {vector_db_path}")
        except Exception as e:
            print(f"⚠️  Failed to save vector database: {e}")


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Autonomous Defect Triage API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - app_state.start_time
    
    return HealthResponse(
        status="healthy" if app_state.is_ready else "starting",
        version="1.0.0",
        vector_db_size=len(app_state.vector_memory) if app_state.vector_memory else 0,
        model_name=app_state.vector_memory.model_name if app_state.vector_memory else "unknown",
        uptime_seconds=uptime,
        timestamp=datetime.now().isoformat()
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats(memory: VectorMemory = Depends(get_vector_memory)):
    """Get system statistics."""
    stats = memory.get_stats()
    
    return StatsResponse(
        total_defects=stats["num_documents"],
        model_name=stats["model_name"],
        index_type=stats["index_type"],
        embedding_dimension=stats["dimension"],
        api_version="1.0.0"
    )


@app.post("/analyze", response_model=TriageResponse)
async def analyze_defect(
    request: AnalyzeRequest,
    agent: DefectTriageAgent = Depends(get_agent)
):
    """
    Analyze a single error log and provide root cause analysis.
    
    **Parameters:**
    - **error_log**: The error log text to analyze
    - **top_k**: Number of similar historical defects to retrieve (1-10)
    - **include_similar**: Whether to include similar defects in response
    
    **Returns:**
    - Root cause analysis with confidence score and recommendations
    """
    start_time = time.time()
    
    try:
        # Analyze defect
        result = await agent.analyze_defect(
            error_log=request.error_log,
            top_k=request.top_k
        )
        
        analysis_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Build response
        similar_defects = None
        if request.include_similar:
            similar_defects = result.similar_defects
        
        return TriageResponse(
            root_cause=result.root_cause,
            confidence=result.confidence,
            reasoning=result.reasoning,
            recommendations=result.recommendations,
            similar_defects=similar_defects,
            analysis_time_ms=analysis_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/analyze/batch", response_model=BatchTriageResponse)
async def batch_analyze_defects(
    request: BatchAnalyzeRequest,
    agent: DefectTriageAgent = Depends(get_agent)
):
    """
    Analyze multiple error logs in batch.
    
    **Parameters:**
    - **error_logs**: List of error log texts (max 50)
    - **top_k**: Number of similar defects per error (1-10)
    
    **Returns:**
    - List of triage results for each error log
    """
    start_time = time.time()
    
    try:
        # Batch analyze
        results = await agent.batch_analyze(
            error_logs=request.error_logs,
            top_k=request.top_k
        )
        
        total_time = (time.time() - start_time) * 1000
        
        # Convert to response format
        triage_responses = []
        for result in results:
            triage_responses.append(TriageResponse(
                root_cause=result.root_cause,
                confidence=result.confidence,
                reasoning=result.reasoning,
                recommendations=result.recommendations,
                similar_defects=result.similar_defects,
                analysis_time_ms=0,  # Individual timing not tracked in batch
                timestamp=datetime.now().isoformat()
            ))
        
        return BatchTriageResponse(
            results=triage_responses,
            total_count=len(results),
            total_time_ms=total_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch analysis failed: {str(e)}"
        )


@app.post("/add-defect", response_model=AddDefectResponse)
async def add_defect_to_knowledge_base(
    request: AddDefectRequest,
    background_tasks: BackgroundTasks,
    memory: VectorMemory = Depends(get_vector_memory)
):
    """
    Add a new defect to the knowledge base.
    
    **Parameters:**
    - **error_log**: The error log text
    - **metadata**: Optional metadata (source, resolution, etc.)
    - **source**: Source system (e.g., Jenkins, GitHub)
    
    **Returns:**
    - Confirmation with number of chunks added
    """
    try:
        # Process log into chunks
        ingestor = LogIngestor()
        chunks = ingestor.process_log_string(request.error_log)
        
        # Add metadata
        metadata_list = []
        for _ in chunks:
            meta = request.metadata.copy() if request.metadata else {}
            if request.source:
                meta["source"] = request.source
            meta["added_at"] = datetime.now().isoformat()
            metadata_list.append(meta)
        
        # Add to vector memory
        chunks_added = await memory.add_documents_async(chunks, metadata_list)
        
        # Save vector memory in background
        import os
        vector_db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")
        background_tasks.add_task(memory.save, vector_db_path)
        
        return AddDefectResponse(
            chunks_added=chunks_added,
            total_defects=len(memory),
            message="Defect added successfully. Knowledge base updated.",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add defect: {str(e)}"
        )


@app.post("/clear-knowledge-base")
async def clear_knowledge_base(
    confirm: bool = False,
    memory: VectorMemory = Depends(get_vector_memory)
):
    """
    Clear the entire knowledge base. Use with caution!
    
    **Parameters:**
    - **confirm**: Must be true to proceed
    
    **Returns:**
    - Confirmation message
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must set confirm=true to clear knowledge base"
        )
    
    try:
        memory.clear()
        
        return {
            "message": "Knowledge base cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear knowledge base: {str(e)}"
        )


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


# Run server
if __name__ == "__main__":
    import os
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
        log_level=os.getenv("API_LOG_LEVEL", "info")
    )
