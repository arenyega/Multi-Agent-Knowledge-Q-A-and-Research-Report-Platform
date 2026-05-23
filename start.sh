#!/bin/bash

# AI Agents Production Startup Script - ModelScope/Text-RAG version

echo "🚀 Starting AI Agents Production System..."
echo "=========================================="

if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create it first:"
    echo ""
    echo "cp env.template .env"
    echo ""
    echo "Then edit .env and set:"
    echo "OPENAI_BASE_URL=https://api-inference.modelscope.cn/v1"
    echo "OPENAI_API_KEY=your_modelscope_openai_api_key"
    echo "LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash"
    echo "TAVILY_API_KEY=your_tavily_api_key"
    echo ""
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running!"
    echo "Please start Docker and try again."
    exit 1
fi

if command -v docker-compose &> /dev/null; then
    COMPOSE="docker-compose"
elif docker compose version > /dev/null 2>&1; then
    COMPOSE="docker compose"
else
    echo "❌ Error: Docker Compose is not installed!"
    echo "Please install Docker Compose and try again."
    exit 1
fi

echo "📁 Creating storage directories..."
mkdir -p src storage/uploads storage/reports storage/qdrant_data

echo "🐳 Building and starting services..."
$COMPOSE down
$COMPOSE pull
$COMPOSE up -d --build

echo "⏳ Waiting for services to start..."
sleep 10

echo "🔍 Checking service health..."

echo "Checking Qdrant database..."
if curl -s http://localhost:6333/health > /dev/null; then
    echo "✅ Qdrant is running"
else
    echo "⚠️  Qdrant may still be starting up..."
fi

echo "Checking Backend API..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend API is running"
else
    echo "⚠️  Backend API may still be starting up..."
fi

echo "Checking Frontend..."
if curl -s http://localhost:8501 > /dev/null; then
    echo "✅ Frontend is running"
else
    echo "⚠️  Frontend may still be starting up..."
fi

echo ""
echo "🎉 AI Agents System is starting up!"
echo "=========================================="
echo "🌐 Frontend UI:    http://localhost:8501"
echo "🔧 Backend API:    http://localhost:8000"
echo "📖 API Docs:       http://localhost:8000/docs"
echo "🗄️  Qdrant DB:     http://localhost:6333/dashboard"
echo ""
echo "📝 To view logs: $COMPOSE logs -f"
echo "🛑 To stop:      $COMPOSE down"
