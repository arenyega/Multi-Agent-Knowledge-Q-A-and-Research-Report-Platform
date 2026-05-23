"""
File handling utilities
"""

import os
import tempfile
import aiofiles
from fastapi import UploadFile

async def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file to temporary location and return path"""
    
    # Create uploads directory if it doesn't exist
    uploads_dir = "storage/uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Generate unique filename
    filename = f"{upload_file.filename}"
    file_path = os.path.join(uploads_dir, filename)
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    
    return file_path

def ensure_directories():
    """Ensure required directories exist"""
    directories = [
        "storage/uploads",
        "storage/reports",
        "storage/qdrant_data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True) 