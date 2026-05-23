"""
Clean service wrapper for Research Agent

This service uses the moved researcher implementation directly,
providing research report generation with FastAPI integration.
"""

import os
import asyncio
import uuid
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the moved researcher components
try:
    from .researcher import build_research_graph, GraphState
    RESEARCHER_AVAILABLE = True
    logger.info("✅ Researcher components loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️  Could not import researcher components: {e}")
    RESEARCHER_AVAILABLE = False
    build_research_graph = None
    GraphState = None


class ResearcherService:
    """Clean service wrapper for the Research agent"""
    
    def __init__(self):
        self.research_graph = None
        self.reports_dir = Path("/app/storage/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._check_dependencies()
        self._initialize_graph()
    
    def _check_dependencies(self):
        """Check if all required dependencies are available"""
        required_env_vars = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "TAVILY_API_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"Missing environment variables: {missing_vars}")
            self.dependencies_available = False
        else:
            self.dependencies_available = True and RESEARCHER_AVAILABLE
        
        logger.info(f"Dependencies available: {self.dependencies_available}")
    
    def _initialize_graph(self):
        """Initialize the research graph"""
        if self.dependencies_available and build_research_graph:
            try:
                self.research_graph = build_research_graph()
                logger.info("✅ Research graph initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing research graph: {e}")
                self.dependencies_available = False

    async def generate_research_report(self, query: str, save_report: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive research report for the given query
        
        Args:
            query: The research question/topic
            save_report: Whether to save the report to storage
            
        Returns:
            Dict containing the report content and metadata
        """
        if not self.dependencies_available or not self.research_graph:
            return self._generate_placeholder_response(query)
        
        try:
            logger.info(f"🔍 Generating research report for: {query}")
            
            # Prepare initial state
            initial_state = {
                "question": query,
                "persona_prompt": "",
                "search_results": [],
                "markdown_answer": ""
            }
            
            # Run the research workflow
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.research_graph.invoke,
                initial_state
            )
            
            # Extract results
            markdown_report = result.get("markdown_answer", "")
            search_results_count = len(result.get("search_results", []))
            persona_used = result.get("persona_prompt", "")
            
            # Generate report metadata
            report_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            report_data = {
                "report_id": report_id,
                "query": query,
                "markdown_content": markdown_report,
                "metadata": {
                    "timestamp": timestamp,
                    "search_results_count": search_results_count,
                    "persona_used": persona_used[:100] + "..." if len(persona_used) > 100 else persona_used,
                    "word_count": len(markdown_report.split()),
                    "character_count": len(markdown_report)
                }
            }
            
            # Save report if requested
            if save_report and markdown_report:
                await self._save_report(report_id, query, markdown_report, timestamp)
            
            logger.info(f"✅ Research report generated successfully (ID: {report_id})")
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating research report: {e}")
            return self._generate_error_response(query, str(e))
    
    async def _save_report(self, report_id: str, query: str, content: str, timestamp: str):
        """Save the research report to storage"""
        try:
            filename = f"{report_id}.md"
            filepath = self.reports_dir / filename
            
            # Create report with metadata header
            report_content = f"""---
title: Research Report
query: "{query}"
report_id: {report_id}
timestamp: {timestamp}
generated_by: AI Research Agent
---

{content}
"""
            
            # Write to file
            await asyncio.get_event_loop().run_in_executor(
                None,
                filepath.write_text,
                report_content,
                "utf-8"
            )
            
            logger.info(f"📄 Report saved: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    def get_report_file_path(self, report_id: str) -> Optional[Path]:
        """Get the file path for a saved report"""
        filepath = self.reports_dir / f"{report_id}.md"
        if filepath.exists():
            return filepath
        return None
    
    def list_saved_reports(self) -> list:
        """List all saved research reports"""
        reports = []
        try:
            for filepath in self.reports_dir.glob("*.md"):
                try:
                    content = filepath.read_text(encoding="utf-8")
                    # Extract metadata from the header
                    if content.startswith("---"):
                        lines = content.split("\n")
                        metadata = {}
                        for line in lines[1:]:
                            if line == "---":
                                break
                            if ":" in line:
                                key, value = line.split(":", 1)
                                metadata[key.strip()] = value.strip().strip('"')
                        
                        reports.append({
                            "report_id": filepath.stem,
                            "filename": filepath.name,
                            "query": metadata.get("query", "Unknown"),
                            "timestamp": metadata.get("timestamp", "Unknown"),
                            "file_size": filepath.stat().st_size
                        })
                except Exception as e:
                    logger.warning(f"Error reading report {filepath.name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error listing reports: {e}")
        
        # Sort by timestamp, newest first
        reports.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return reports
    

    
    def _generate_placeholder_response(self, query: str) -> Dict[str, Any]:
        """Generate placeholder response when dependencies are not available"""
        report_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        placeholder_content = f"""# Research Report: {query}

**Status:** Placeholder Mode - API Keys Required

## What Would Happen

If the required API keys were configured, this research agent would:

1. **Task Processing**: Parse and understand your research query
2. **Planner**: Generate a specialized researcher persona for your topic
3. **Researcher**: Create 4 targeted search queries and execute web searches using Tavily
4. **Publisher**: Synthesize all findings into a comprehensive markdown report

## Required Configuration

To enable full research functionality:
- Set OPENAI_API_KEY in your .env file (ModelScope/OpenAI-compatible API)
- Set OPENAI_BASE_URL in your .env file
- Set TAVILY_API_KEY in your .env file (for web search)

## System Status

✅ Research workflow architecture ready  
✅ FastAPI integration functional  
✅ Report generation and storage system operational  
⚠️ Waiting for API key configuration

*Your query "{query}" is ready to be processed once the AI services are configured.*
"""
        
        return {
            "report_id": report_id,
            "query": query,
            "markdown_content": placeholder_content,
            "metadata": {
                "timestamp": timestamp,
                "search_results_count": 0,
                "persona_used": "Placeholder mode",
                "word_count": len(placeholder_content.split()),
                "character_count": len(placeholder_content),
                "status": "placeholder"
            }
        }
    
    def _generate_error_response(self, query: str, error: str) -> Dict[str, Any]:
        """Generate error response"""
        report_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        error_content = f"""# Research Report Error

**Query:** {query}  
**Error:** {error}  
**Timestamp:** {timestamp}

## What Happened

The research agent encountered an issue while processing your query. This might be due to:

- API rate limits or connectivity issues
- Invalid search queries or topics
- Temporary service unavailability
- Configuration problems

## Troubleshooting

1. **Check API Keys**: Ensure OPENAI_API_KEY, OPENAI_BASE_URL, and TAVILY_API_KEY are valid
2. **Rephrase Query**: Try a different phrasing of your research question
3. **Wait and Retry**: Some issues resolve automatically after a short wait
4. **Check Logs**: Review the system logs for detailed error information

## Support

If the issue persists, please check the application logs or contact support with this report ID: `{report_id}`
"""
        
        return {
            "report_id": report_id,
            "query": query,
            "markdown_content": error_content,
            "metadata": {
                "timestamp": timestamp,
                "search_results_count": 0,
                "persona_used": "Error occurred",
                "word_count": len(error_content.split()),
                "character_count": len(error_content),
                "status": "error",
                "error": error
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed service status"""
        return {
            "researcher_available": RESEARCHER_AVAILABLE,
            "dependencies_available": self.dependencies_available,
            "graph_initialized": self.research_graph is not None,
            "reports_directory": str(self.reports_dir),
            "saved_reports_count": len(list(self.reports_dir.glob("*.md")))
        } 