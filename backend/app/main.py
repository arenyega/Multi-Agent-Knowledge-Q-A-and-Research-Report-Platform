"""
FastAPI Backend for AI Agents - Clean Version

Serves the sophisticated Agentic RAG agent through REST API endpoints.
Supports session management for follow-up questions and real-time progress streaming.
"""

import os
import tempfile
import asyncio
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import logging

# Set up logging
logger = logging.getLogger(__name__)

from app.agents.agentic_rag_service import AgenticRAGService
from app.agents.researcher_service import ResearcherService
from app.utils.file_utils import save_uploaded_file
from app.models.schemas import (
    ChatRequest, 
    ChatResponse, 
    UploadResponse,
    StatusResponse
)

# Global services
agentic_rag_service = None
researcher_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup services"""
    global agentic_rag_service, researcher_service
    
    # Initialize enhanced services
    agentic_rag_service = AgenticRAGService()
    researcher_service = ResearcherService()

    yield
    
    # Cleanup if needed
    pass

app = FastAPI(
    title="AI Agents Backend - Enhanced",
    description="Backend API for sophisticated Agentic RAG and Research Agent with comprehensive report generation",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Agents Backend is running!",
        "version": "2.0.0 - Enhanced",
        "agents": {
            "agentic_rag": "PDF text extraction with text embeddings and optional web search",
            "researcher": "Comprehensive research report generation with web search"
        },
        "features": [
            "Sophisticated Agentic RAG with multimodal processing",
            "Research agent with web search and report generation", 
            "Session management for follow-up questions",
            "Graph-based agent architectures",
            "Report storage and download functionality"
        ]
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    rag_status = agentic_rag_service.get_status() if agentic_rag_service else {}
    researcher_status = researcher_service.get_status() if researcher_service else {}
    
    return {
        "status": "healthy",
        "message": "All services are operational",
        "services": {
            "agentic_rag": rag_status,
            "researcher": researcher_status
        }
    }

# Enhanced Agentic RAG Endpoints
@app.post("/api/agentic-rag/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), session_id: Optional[str] = Query(default="default")):
    """Upload and process PDF for sophisticated RAG system with session support"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file
        file_path = await save_uploaded_file(file)
        
        # Process PDF through sophisticated ingestion with session support
        success = await agentic_rag_service.ingest_pdf(file_path, session_id)
        
        if success:
            return UploadResponse(
                success=True,
                message=f"PDF '{file.filename}' uploaded and processed with sophisticated multimodal ingestion",
                filename=file.filename,
                details={
                    "session_id": session_id,
                    "processing_type": "multimodal_image_extraction",
                    "features_enabled": ["document_grading", "web_search_routing", "hallucination_detection"]
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to process PDF")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/agentic-rag/chat", response_model=ChatResponse)
async def chat_with_pdf(request: ChatRequest, session_id: Optional[str] = Query(default="default")):
    """Chat with uploaded PDF using sophisticated agentic RAG with session support for follow-up questions"""
    try:
        response = await agentic_rag_service.query(request.message, session_id)
        return ChatResponse(
            response=response,
            success=True,
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/api/agentic-rag/history/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    try:
        history = agentic_rag_service.get_conversation_history(session_id)
        return {
            "session_id": session_id,
            "conversation_history": history,
            "total_exchanges": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@app.delete("/api/agentic-rag/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    try:
        success = agentic_rag_service.clear_session(session_id)
        if success:
            return {"message": f"Session {session_id} cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")

# Research Agent Endpoints
@app.post("/api/researcher/generate-report")
async def generate_research_report(request: ChatRequest, save_report: bool = Query(default=True)):
    """Generate a comprehensive research report on the given topic"""
    try:
        result = await researcher_service.generate_research_report(request.message, save_report)
        return {
            "success": True,
            "report_id": result["report_id"],
            "query": result["query"],
            "markdown_content": result["markdown_content"],
            "metadata": result["metadata"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research report generation failed: {str(e)}")

@app.get("/api/researcher/reports")
async def list_research_reports():
    """List all saved research reports"""
    try:
        reports = researcher_service.list_saved_reports()
        return {
            "success": True,
            "reports": reports,
            "total_reports": len(reports)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")

@app.get("/api/researcher/reports/{report_id}")
async def get_research_report(report_id: str):
    """Get a specific research report by ID"""
    try:
        file_path = researcher_service.get_report_file_path(report_id)
        if file_path and file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            return {
                "success": True,
                "report_id": report_id,
                "content": content,
                "filename": file_path.name
            }
        else:
            raise HTTPException(status_code=404, detail="Report not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get report: {str(e)}")

@app.get("/api/researcher/reports/{report_id}/download")
async def download_research_report(report_id: str):
    """Download a research report as a markdown file"""
    try:
        file_path = researcher_service.get_report_file_path(report_id)
        if file_path and file_path.exists():
            return FileResponse(
                path=str(file_path),
                media_type='text/markdown',
                filename=f"research_report_{report_id}.md",
                headers={"Content-Disposition": f"attachment; filename=research_report_{report_id}.md"}
            )
        else:
            raise HTTPException(status_code=404, detail="Report not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download report: {str(e)}")

@app.get("/api/researcher/status")
async def get_researcher_status():
    """Get detailed status of the researcher service"""
    try:
        status = researcher_service.get_status()
        return {
            "success": True,
            "status": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 