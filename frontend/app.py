"""
Streamlit Frontend for AI Agents

Provides a chat interface for both Agentic RAG and Researcher agents.
"""

import streamlit as st
import requests
import time
import os
from typing import Optional

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def init_session_state():
    """Initialize session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_agent" not in st.session_state:
        st.session_state.current_agent = "Chat with your PDF"
    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = False
    if "research_reports" not in st.session_state:
        st.session_state.research_reports = []
    if "selected_report" not in st.session_state:
        st.session_state.selected_report = None

def upload_pdf(uploaded_file) -> bool:
    """Upload PDF to backend"""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post(f"{BACKEND_URL}/api/agentic-rag/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            st.success(result["message"])
            return True
        else:
            st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
            return False
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return False

def chat_with_pdf(message: str) -> Optional[str]:
    """Send message to agentic RAG agent"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/agentic-rag/chat",
            json={"message": message}
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["response"]
        else:
            st.error(f"Chat failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Chat error: {str(e)}")
        return None

def generate_research_report(query: str) -> Optional[dict]:
    """Generate research report"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/researcher/generate-report",
            json={"message": query}
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            st.error(f"Research generation failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Research error: {str(e)}")
        return None

def list_research_reports() -> list:
    """List all research reports"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/researcher/reports")
        
        if response.status_code == 200:
            result = response.json()
            return result.get("reports", [])
        else:
            return []
    except Exception as e:
        st.error(f"Failed to list reports: {str(e)}")
        return []

def download_research_report(report_id: str, filename: str = None):
    """Download research report"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/researcher/reports/{report_id}/download")
        
        if response.status_code == 200:
            if not filename:
                filename = f"research_report_{report_id}.md"
            
            st.download_button(
                label="📄 Download Research Report",
                data=response.content,
                file_name=filename,
                mime="text/markdown"
            )
        else:
            st.error("Failed to download report")
    except Exception as e:
        st.error(f"Download error: {str(e)}")

def render_agentic_rag_interface():
    """Render the Agentic RAG chat interface"""
    st.header("🔍 Chat with your PDF")
    st.write("Upload a PDF document and ask questions about its content.")
    
    # PDF Upload Section
    if not st.session_state.pdf_uploaded:
        st.subheader("📁 Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            if st.button("Upload and Process PDF"):
                with st.spinner("Processing PDF..."):
                    if upload_pdf(uploaded_file):
                        st.session_state.pdf_uploaded = True
                        st.rerun()
    else:
        st.success("✅ PDF uploaded and processed successfully!")
        if st.button("Upload New PDF"):
            st.session_state.pdf_uploaded = False
            st.session_state.chat_history = []
            st.rerun()
    
    # Chat Interface
    if st.session_state.pdf_uploaded:
        st.subheader("💬 Ask Questions")
        
        # Display chat history
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.chat_message("user").write(chat["content"])
            else:
                st.chat_message("assistant").write(chat["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your PDF..."):
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            # Get response from backend
            with st.spinner("Thinking..."):
                response = chat_with_pdf(prompt)
                
                if response:
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.chat_message("assistant").write(response)
                else:
                    st.error("Failed to get response")

def render_researcher_interface():
    """Render the Researcher interface"""
    st.header("🔬 Deep Research")
    st.write("Enter a research topic and get a comprehensive markdown report.")
    
    # Create tabs for Generate and View Reports
    tab1, tab2 = st.tabs(["Generate New Report", "View Saved Reports"])
    
    with tab1:
        # Research Input
        research_query = st.text_area(
            "Research Topic", 
            placeholder="Enter your research question or topic...",
            help="Be specific about what you want to research",
            height=100
        )
        
        if st.button("🚀 Generate Research Report", type="primary"):
            if research_query.strip():
                with st.spinner("Generating comprehensive research report... This may take 30-60 seconds."):
                    result = generate_research_report(research_query)
                    if result:
                        st.success("✅ Research report generated successfully!")
                        
                        # Display report metadata
                        metadata = result.get("metadata", {})
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Word Count", metadata.get("word_count", 0))
                        with col2:
                            st.metric("Search Results", metadata.get("search_results_count", 0))
                        with col3:
                            st.metric("Character Count", metadata.get("character_count", 0))
                        with col4:
                            timestamp = metadata.get("timestamp", "")
                            if timestamp:
                                try:
                                    from datetime import datetime
                                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                    formatted_time = dt.strftime("%H:%M:%S")
                                    st.metric("Generated At", formatted_time)
                                except:
                                    st.metric("Generated", "Just now")
                        
                        # Display the report content
                        st.subheader("📄 Generated Report")
                        st.markdown(result.get("markdown_content", ""))
                        
                        # Download button
                        if result.get("report_id"):
                            download_research_report(result["report_id"])
                        
                        # Update the saved reports list
                        st.session_state.research_reports = list_research_reports()
                        
            else:
                st.error("Please enter a research topic")
    
    with tab2:
        st.subheader("📚 Saved Research Reports")
        
        # Refresh button
        if st.button("🔄 Refresh Reports"):
            st.session_state.research_reports = list_research_reports()
        
        # Load reports if not already loaded
        if not st.session_state.research_reports:
            st.session_state.research_reports = list_research_reports()
        
        if st.session_state.research_reports:
            # Display reports in a table format
            for report in st.session_state.research_reports:
                with st.expander(f"📄 {report.get('query', 'Unknown Query')}", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Report ID:** `{report.get('report_id', 'N/A')}`")
                        st.write(f"**Timestamp:** {report.get('timestamp', 'N/A')}")
                        st.write(f"**File Size:** {report.get('file_size', 0)} bytes")
                    
                    with col2:
                        if st.button(f"📥 Download", key=f"download_{report.get('report_id')}"):
                            download_research_report(
                                report.get('report_id'), 
                                report.get('filename', f"report_{report.get('report_id')}.md")
                            )
        else:
            st.info("No research reports found. Generate your first report in the 'Generate New Report' tab!")

def main():
    """Main application"""
    st.set_page_config(
        page_title="AI Agents Chat",
        page_icon="🤖",
        layout="wide"
    )
    
    init_session_state()
    
    # Sidebar for agent selection
    with st.sidebar:
        st.title("🤖 AI Agents")
        st.write("Select an agent to interact with:")
        
        agent_choice = st.radio(
            "Choose Agent:",
            ["Chat with your PDF", "Deep Research"],
            index=0 if st.session_state.current_agent == "Chat with your PDF" else 1
        )
        
        if agent_choice != st.session_state.current_agent:
            st.session_state.current_agent = agent_choice
            st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        if agent_choice == "Chat with your PDF":
            st.markdown("""
            **Agentic RAG**
            - Upload PDF documents
            - Ask questions about content
            - Get intelligent answers
            - Supports image-based PDFs
            """)
        else:
            st.markdown("""
            **Research Agent**
            - Enter any research topic
            - Get comprehensive reports
            - Multiple web searches
            - Downloadable markdown
            """)
    
    # Main content area
    if st.session_state.current_agent == "Chat with your PDF":
        render_agentic_rag_interface()
    else:
        render_researcher_interface()

if __name__ == "__main__":
    main() 