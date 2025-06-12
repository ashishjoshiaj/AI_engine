import os
import json
import re
import logging
import asyncio
from datetime import datetime
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from questions import empirical_questions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure API key exists
if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("Missing OPENAI_API_KEY in environment variables.")

llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.5,
    max_retries=2,
    request_timeout=30,  # Timeout per request
    max_tokens=500       # Limit response length
)

def chunk_text(text: str, chunk_size: int = 3000, chunk_overlap: int = 150) -> list:
    """Optimized text chunking"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
        keep_separator=True
    )
    return text_splitter.split_text(text)

async def ask_llm(content: str, question: str, document_type: str = "study") -> dict:
    """Optimized LLM interaction with timeout handling"""
    # Simplified system prompts
    system_prompt = {
        "ipcc report": "You're analyzing an IPCC climate assessment. Focus on methodology.",
        "preprint": "You're analyzing a scientific preprint. Note it hasn't been peer-reviewed.",
        "peer_reviewed": "You're evaluating a peer-reviewed climate study.",
        "study": "You're evaluating a climate science study."
    }.get(document_type, "You're evaluating a scientific document.")
    
    # Content excerpt (shorter)
    content_excerpt = content[:2000] + "..." if len(content) > 2000 else content
    
    user_prompt = f"""**Analysis Question**:
{question}

**Document Excerpt**:
{content_excerpt}

**Response Format (JSON)**:
{{
  "score": <1-10>,
  "justification": "<1 sentence>"
}}"""

    messages = [
        SystemMessage(content=system_prompt + "\n\nRespond ONLY with valid JSON."),
        HumanMessage(content=user_prompt)
    ]

    try:
        # Use async call with timeout
        response = await asyncio.wait_for(
            llm.agenerate([messages]),
            timeout=20.0
        )
        raw = response.generations[0][0].text.strip()

        # Simplified JSON parsing
        try:
            if raw.startswith("```json"):
                raw = raw[7:].strip().strip("`")
            elif raw.startswith("```"):
                raw = raw[3:].strip().strip("`")
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fallback to regex parsing
            score_match = re.search(r'"score":\s*([\d.]+)', raw)
            justification_match = re.search(r'"justification":\s*"([^"]+)"', raw)
            if score_match and justification_match:
                return {
                    "score": float(score_match.group(1)),
                    "justification": justification_match.group(1)
                }
            raise

    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"LLM timeout/error: {str(e)[:100]}")
        return {
            "score": 5.0,
            "justification": "Analysis timed out"
        }

def determine_document_type(source: str, url: str) -> str:
    """Faster document type detection without content analysis"""
    source_lower = source.lower()
    url_lower = url.lower()
    
    if "ipcc" in url_lower or "ipcc" in source_lower:
        return "ipcc report"
    if "arxiv" in url_lower:
        return "preprint"
    if any(j in source_lower for j in ["nature", "science", "journal"]):
        return "peer_reviewed"
    return "study"

async def analyze_study(content: str, url: str, title: str, authors: str, source: str) -> dict:
    """Optimized analysis function with timeout handling"""
    # Validate content
    if len(content) < 1000:
        raise ValueError("Document content too short for analysis")
    
    # Faster document typing
    doc_type = determine_document_type(source, url)
    processing_notes = [f"Document length: {len(content):,} chars"]
    
    # Prepare all questions
    tasks = []
    for category, questions in empirical_questions.items():
        for question in questions:
            tasks.append((category, question, ask_llm(content, question, doc_type)))
    
    # Process questions concurrently with timeout
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*(t[2] for t in tasks)),
            timeout=120.0  # Overall timeout (2 minutes)
        )
    except asyncio.TimeoutError:
        logger.error("Overall analysis timed out")
        raise HTTPException(status_code=504, detail="Analysis timed out")
    
    # Organize results
    analysis = {}
    all_scores = []
    for idx, (category, question, _) in enumerate(tasks):
        result = results[idx]
        if category not in analysis:
            analysis[category] = {"questions": [], "scores": []}
        
        analysis[category]["questions"].append({
            "question": question,
            "score": result["score"],
            "justification": result["justification"]
        })
        analysis[category]["scores"].append(result["score"])
        all_scores.append(result["score"])
    
    # Calculate scores
    for category in analysis:
        scores = analysis[category]["scores"]
        analysis[category]["subscore"] = round(sum(scores) / len(scores), 2)
        del analysis[category]["scores"]  # Cleanup temporary field
    
    # Final scoring
    final_es_score = round(sum(all_scores) / len(all_scores), 2)
    empirical_score = analysis.get("empirical_basis", {}).get("subscore", 5.0)
    modeled_score = analysis.get("modeled_science", {}).get("subscore", 5.0)
    
    if empirical_score >= 7:
        classification = "Tier 1 (Empirical Science)"
        summary = "Primarily empirical and data-driven"
    elif modeled_score >= 7:
        classification = "Tier 2 (Modeled Science)"
        summary = "Relies on simulations and models"
    else:
        classification = "Tier 3 (Scenario Analysis)"
        summary = "Focuses on scenarios and narratives"
    
    return {
        "title": title,
        "authors": authors,
        "source": source,
        "url": url,
        "document_type": doc_type,
        "analysis": {
            "empirical_basis": analysis.get("empirical_basis", {"questions": [], "subscore": 5.0}),
            "modeled_science": analysis.get("modeled_science", {"questions": [], "subscore": 5.0}),
            "storyline_science": analysis.get("storyline_science", {"questions": [], "subscore": 5.0}),
            "final_es_score": final_es_score,
            "classification": classification,
            "summary": summary
        },
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "processing_notes": processing_notes
    }