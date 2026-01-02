#!/usr/bin/env python3
"""
Demo script for the Multi-Agent Research Orchestrator.

This script demonstrates the multi-agent system working together
to research a topic, without needing the full API server.

Usage:
    python demo.py "What are the latest developments in quantum computing?"
    python demo.py --mock "Explain the impact of AI on healthcare"
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

# Load environment
try:
    from dotenv import load_dotenv
    from pathlib import Path
    # Check parent directory first (project root)
    parent_env = Path(__file__).parent.parent / ".env"
    if parent_env.exists():
        load_dotenv(parent_env)
    else:
        load_dotenv()
except ImportError:
    pass


class MockLLMClient:
    """Mock LLM client for demo without API keys."""
    
    class ChatCompletions:
        async def create(self, **kwargs):
            messages = kwargs.get("messages", [])
            user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
            
            # Simulate different agent responses based on system prompt
            if "Research Agent" in system_msg or "web_search" in str(messages):
                response = json.dumps({
                    "sources": [
                        {"title": "Academic Research Paper on the Topic", "url": "https://arxiv.org/example", "snippet": "Key findings from peer-reviewed research...", "relevance": 0.95},
                        {"title": "Industry Analysis Report 2024", "url": "https://industry.example.com", "snippet": "Market trends and analysis from leading experts...", "relevance": 0.88},
                        {"title": "Government Policy Document", "url": "https://gov.example.org", "snippet": "Official guidelines and regulatory framework...", "relevance": 0.75}
                    ],
                    "key_findings": [
                        "Significant progress has been made with measurable outcomes",
                        "Multiple stakeholders are investing heavily in development",
                        "Practical implementation challenges remain to be addressed",
                        "Cross-industry collaboration is accelerating innovation"
                    ],
                    "confidence": 0.85,
                    "search_queries_used": [user_msg[:50]]
                })
            elif "Analysis Agent" in system_msg or "Analyze" in user_msg:
                response = json.dumps({
                    "summary": "This comprehensive analysis reveals that the field is experiencing rapid transformation. Academic research and industry developments are converging to create practical solutions. Key stakeholders from technology, government, and research sectors are actively contributing to advancement. While significant progress has been made, challenges in standardization and widespread adoption remain. The trajectory suggests continued growth with increasing real-world applications expected over the coming years.",
                    "key_points": [
                        "Rapid technological advancement with breakthrough innovations",
                        "Major global companies investing billions in R&D",
                        "Regulatory frameworks evolving to address new challenges",
                        "Growing public awareness driving market demand",
                        "Cross-sector collaboration accelerating development"
                    ],
                    "patterns": [
                        {"pattern": "Accelerating innovation cycle", "evidence": ["Multiple papers published monthly", "Increased VC funding"], "significance": "Indicates maturing field"}
                    ],
                    "themes": ["Innovation", "Investment", "Regulation", "Global Adoption"],
                    "gaps": ["Long-term impact studies needed", "Standardization efforts required", "Workforce training programs lacking"]
                })
            elif "Critic Agent" in system_msg or "critique" in user_msg.lower():
                response = json.dumps({
                    "overall_confidence": 0.78,
                    "verified_claims": [
                        {"claim": "Technology is advancing rapidly", "confidence": 0.92, "evidence": "Multiple peer-reviewed sources confirm"},
                        {"claim": "Major investments being made", "confidence": 0.88, "evidence": "Financial reports from leading companies"}
                    ],
                    "unverified_claims": [
                        {"claim": "Will revolutionize industry by 2025", "reason": "Speculative timeline - actual adoption rates vary"}
                    ],
                    "contradictions": [],
                    "potential_biases": [
                        {"type": "Source bias", "description": "Majority of sources from industry stakeholders", "severity": "low"}
                    ],
                    "source_assessment": {"reliability": 0.85, "diversity": 0.72, "recency": 0.90},
                    "speculation_flags": ["Future predictions inherently uncertain", "Market projections may be optimistic"],
                    "improvement_suggestions": ["Include more independent academic sources", "Add historical context for comparison"],
                    "final_verdict": "The research findings are generally reliable with good source diversity. Claims are well-supported by evidence with appropriate confidence levels assigned."
                })
            elif "Synthesize" in user_msg or "final report" in user_msg.lower():
                # Final synthesis from supervisor
                response = """Based on comprehensive research and analysis, here are the key findings:

**Executive Summary**
The research reveals significant developments in this field, with strong evidence of advancement across multiple dimensions. Our multi-agent analysis achieved a confidence score of 78%, indicating reliable findings with some areas requiring further investigation.

**Key Findings**
1. Technology is advancing rapidly with measurable breakthroughs
2. Major stakeholders are investing significant resources
3. Regulatory frameworks are evolving to address new challenges
4. Public awareness and adoption are increasing globally

**Confidence Assessment**
- Overall reliability: High (78%)
- Source diversity: Good
- Potential biases: Industry stakeholder concentration (low severity)

**Recommendations**
- Monitor emerging developments in this space
- Consider additional independent academic sources for deeper analysis
- Track regulatory changes that may impact the field

This report was generated by a multi-agent research system with cross-validation."""
            else:
                response = f"Processed: {user_msg[:100]}..."
            
            return type('Response', (), {
                'choices': [type('Choice', (), {
                    'message': type('Message', (), {'content': response, 'tool_calls': None})()
                })()]
            })()
    
    def __init__(self):
        self.chat = type('Chat', (), {'completions': self.ChatCompletions()})()


async def run_demo(query: str, use_mock: bool = False):
    """Run the research orchestrator demo."""
    
    print("\n" + "="*60)
    print("🔬 MULTI-AGENT RESEARCH ORCHESTRATOR DEMO")
    print("="*60)
    print(f"\n📝 Query: {query}\n")
    
    # Import agents
    from core.types import AgentRole, ExecutionMode
    from agents.research_agent import ResearchAgent, ResearchAgentConfig
    from agents.analysis_agent import AnalysisAgent, AnalysisAgentConfig
    from agents.critic_agent import CriticAgent, CriticAgentConfig
    from agents.supervisor_agent import SupervisorAgent, SupervisorConfig
    
    # Create LLM client
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if use_mock or (not anthropic_key and not openai_key):
        print("ℹ️  Using mock LLM client (no API key found)\n")
        llm_client = MockLLMClient()
        tavily_client = None
        model = "mock-model"
    else:
        from core.llm import create_llm_client, LLMProvider
        
        if anthropic_key:
            print("✅ Using Claude (Anthropic) as LLM provider\n")
            llm_client = create_llm_client(LLMProvider.ANTHROPIC)
            model = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
        else:
            print("✅ Using OpenAI as LLM provider\n")
            llm_client = create_llm_client(LLMProvider.OPENAI)
            model = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
        
        # Try to get Tavily client
        tavily_client = None
        if os.getenv("TAVILY_API_KEY"):
            try:
                from tavily import TavilyClient
                tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            except ImportError:
                pass
    
    # Create agents
    print("🤖 Initializing agents...")
    
    research_agent = ResearchAgent(
        config=ResearchAgentConfig(
            name="Research Agent",
            description="Gathers information from web searches",
            role=AgentRole.RESEARCH,
            model=model,
            max_search_results=5
        ),
        llm_client=llm_client,
        tavily_client=tavily_client
    )
    
    analysis_agent = AnalysisAgent(
        config=AnalysisAgentConfig(
            name="Analysis Agent",
            description="Summarizes and extracts insights",
            role=AgentRole.ANALYSIS,
            model=model
        ),
        llm_client=llm_client
    )
    
    critic_agent = CriticAgent(
        config=CriticAgentConfig(
            name="Critic Agent",
            description="Fact-checks and assigns confidence",
            role=AgentRole.CRITIC,
            model=model
        ),
        llm_client=llm_client
    )
    
    supervisor = SupervisorAgent(
        config=SupervisorConfig(
            name="Supervisor Agent",
            description="Orchestrates the research workflow",
            role=AgentRole.SUPERVISOR,
            model=model,
            execution_mode=ExecutionMode.SEQUENTIAL,
            max_retries=3,
            enable_human_checkpoints=False
        ),
        llm_client=llm_client,
        research_agent=research_agent,
        analysis_agent=analysis_agent,
        critic_agent=critic_agent
    )
    
    print("   ✓ Research Agent ready")
    print("   ✓ Analysis Agent ready")
    print("   ✓ Critic Agent ready")
    print("   ✓ Supervisor Agent ready\n")
    
    # Execute research
    print("🚀 Starting research pipeline...\n")
    start_time = datetime.now()
    
    print("   [1/4] 📋 Supervisor routing task...")
    print("   [2/4] 🔍 Research Agent searching...")
    
    response = await supervisor.process({"query": query})
    
    print("   [3/4] 📊 Analysis Agent processing...")
    print("   [4/4] ⚖️  Critic Agent evaluating...")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\n✅ Pipeline complete in {elapsed:.1f}s\n")
    print("-"*60)
    
    if response.success:
        try:
            result = json.loads(response.content)
            
            print("\n📊 RESULTS\n")
            
            # Summary
            if result.get("summary"):
                print("📝 Summary:")
                print(f"   {result['summary'][:500]}...\n" if len(result.get('summary', '')) > 500 else f"   {result['summary']}\n")
            
            # Key Points
            if result.get("key_points"):
                print("🎯 Key Points:")
                for i, point in enumerate(result["key_points"][:5], 1):
                    print(f"   {i}. {point}")
                print()
            
            # Confidence
            confidence = result.get("confidence_score", 0)
            conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            print(f"📈 Confidence: [{conf_bar}] {confidence*100:.0f}%\n")
            
            # Citations
            if result.get("citations"):
                print("📚 Sources:")
                for cite in result["citations"][:3]:
                    print(f"   • {cite.get('title', 'Unknown')} ({cite.get('relevance_score', 0)*100:.0f}% relevant)")
                print()
            
            # Contradictions
            if result.get("contradictions"):
                print("⚠️  Contradictions Found:")
                for c in result["contradictions"]:
                    print(f"   • {c}")
                print()
            
            # Agents used
            if result.get("agents_used"):
                print(f"🤖 Agents Used: {' → '.join(result['agents_used'])}")
            
        except json.JSONDecodeError:
            print("📄 Raw Result:")
            print(response.content)
    else:
        print(f"❌ Error: {response.error}")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Research Orchestrator Demo")
    parser.add_argument("query", nargs="?", default="What are the latest developments in artificial intelligence?",
                        help="Research query to investigate")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM (no API key needed)")
    args = parser.parse_args()
    
    asyncio.run(run_demo(args.query, args.mock))


if __name__ == "__main__":
    main()

