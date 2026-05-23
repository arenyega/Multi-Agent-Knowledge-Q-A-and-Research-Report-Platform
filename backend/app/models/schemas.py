"""
Pydantic models for API request/response schemas - Enhanced Version
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel

# Common schemas
class StatusResponse(BaseModel):
    status: str
    message: str
    services: Optional[Dict[str, Any]] = None

# Enhanced Agentic RAG schemas
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    success: bool
    session_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class UploadResponse(BaseModel):
    success: bool
    message: str
    filename: str
    session_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ConversationHistory(BaseModel):
    session_id: str
    conversation_history: List[Dict[str, Any]]
    total_exchanges: int



# Service status schemas
class ServiceStatus(BaseModel):
    initialized: bool
    dependencies_available: bool
    features_available: List[str]
    missing_dependencies: List[str]
    additional_info: Optional[Dict[str, Any]] = None 