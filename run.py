#!/usr/bin/env python3
"""
Run the Research Orchestrator API server.

Usage:
    python run.py                    # Run on default port 8000
    python run.py --port 8080        # Run on custom port
    python run.py --reload           # Run with hot reload (dev mode)

Environment Variables (set in .env file or export):
    ANTHROPIC_API_KEY=sk-ant-...    # Primary: Your Anthropic/Claude API key
    OPENAI_API_KEY=sk-...           # Fallback: OpenAI API key (if no Anthropic)
    TAVILY_API_KEY=tvly-...         # Optional: For web search (uses mock if not set)
    LLM_MODEL=claude-sonnet-4-20250514         # Optional: Model to use

Quick Start:
    1. Create a .env file with your API keys
    2. Install dependencies: pip install -r requirements.txt
    3. Run the server: python run.py
    4. Open http://localhost:8000 in your browser
"""

import os
import sys
import argparse
from pathlib import Path

# Add the research_orchestrator directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Check parent directory first (project root)
    parent_env = Path(__file__).parent.parent / ".env"
    if parent_env.exists():
        load_dotenv(parent_env)
        print(f"✅ Loaded .env from {parent_env}")
    else:
        # Fallback to current directory
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(description="Run the Research Orchestrator API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    # Check for required API keys
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not anthropic_key and not openai_key:
        print("⚠️  Warning: No LLM API key found. The API will not function properly.")
        print("   Set ANTHROPIC_API_KEY (recommended) or OPENAI_API_KEY in your environment.")
    elif anthropic_key:
        print("✅ Using Claude (Anthropic) as LLM provider")
    else:
        print("✅ Using OpenAI as LLM provider")
    
    if not os.getenv("TAVILY_API_KEY"):
        print("ℹ️  Note: TAVILY_API_KEY not set. Web search will use mock data.")

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         Multi-Agent Research Orchestrator                     ║
╠══════════════════════════════════════════════════════════════╣
║  🔬 Research Agent    - Web search & source gathering         ║
║  📊 Analysis Agent    - Summarization & pattern extraction    ║
║  ⚖️  Critic Agent      - Fact-checking & confidence scoring   ║
║  📋 Supervisor Agent  - Workflow coordination & retry logic   ║
╚══════════════════════════════════════════════════════════════╝

🚀 Starting server at http://{args.host}:{args.port}
📖 API docs at http://localhost:{args.port}/docs
🌐 UI at http://localhost:{args.port}
""")

    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()

