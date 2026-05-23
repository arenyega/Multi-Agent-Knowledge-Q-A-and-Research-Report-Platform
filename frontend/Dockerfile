FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8501

# Start Streamlit
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"] 