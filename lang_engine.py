import os
import json
import re
import logging
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

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5, max_retries=3)

def chunk_text(text: str, chunk_size: int = 4000, chunk_overlap: int = 200) -> list:
    """Improved text chunking with better context preservation"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
        keep_separator=True
    )
    return text_splitter.split_text(text)

def ask_llm(content: str, question: str, document_type: str = "study") -> dict:
    """Enhanced LLM interaction with better error handling and prompts"""
    # System prompts tailored to document types
    system_prompts = {
        "ipcc report": "You are Unit 007, analyzing an IPCC climate assessment report. "
                       "Focus on methodology, data sources, and modeling approaches.",
        "preprint": "You are Unit 007, analyzing a scientific preprint. "
                    "Note that this hasn't undergone peer review yet.",
        "peer_reviewed": "You are Unit 007, evaluating a peer-reviewed climate study. "
                         "Assess empirical basis, modeling techniques, and conclusions.",
        "study": "You are Unit 007, evaluating a climate science study. "
                 "Analyze methodology, data quality, and evidence strength."
    }
    
    system_prompt = system_prompts.get(document_type, system_prompts["study"])
    
    # Improved user prompt structure
    user_prompt = f"""**Document Excerpt** (first 4000 chars):
{content[:4000]}

**Analysis Question**:
{question}

**Required Response Format** (JSON ONLY):
{{
  "score": <number 1-10 where 1=weak/no evidence, 10=strong/direct evidence>,
  "justification": "<concise 1-2 sentence explanation>"
}}"""

    messages = [
        SystemMessage(content=system_prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON."),
        HumanMessage(content=user_prompt)
    ]

    try:
        response = llm(messages)
        raw = response.content.strip()

        # Handle JSON wrapped in markdown code blocks
        if raw.startswith("```json"):
            raw = raw[7:].strip().strip("`")
        elif raw.startswith("```"):
            raw = raw[3:].strip().strip("`")

        # Parse JSON with fallback pattern matching
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # Robust pattern matching fallback
            score_match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', raw, re.IGNORECASE)
            justification_match = re.search(r'"justification"\s*:\s*"((?:\\"|[^"])*)"', raw, re.IGNORECASE)
            
            if score_match and justification_match:
                result = {
                    "score": float(score_match.group(1)),
                    "justification": justification_match.group(1)
                }
            else:
                raise ValueError(f"Couldn't parse LLM response: {raw[:200]}...")

        # Validate score range
        if not (1 <= result["score"] <= 10):
            raise ValueError(f"Score {result['score']} out of range (1-10)")
            
        return result

    except Exception as e:
        logger.error(f"LLM query failed: {str(e)}")
        # Return neutral score on failure
        return {
            "score": 5.0,
            "justification": f"Analysis failed: {str(e)[:100]}"
        }

def ask_llm_with_chunks(content: str, question: str, document_type="study") -> dict:
    """Process large documents in chunks with weighted scoring"""
    chunks = chunk_text(content)
    if not chunks:
        return {
            "score": 5.0,
            "justification": "Document too short for analysis"
        }
    
    results = []
    for i, chunk in enumerate(chunks[:10]):  # Limit to first 10 chunks
        try:
            result = ask_llm(chunk, question, document_type)
            results.append(result)
        except Exception as e:
            logger.warning(f"Chunk {i+1} failed: {str(e)}")
    
    if not results:
        return {
            "score": 5.0,
            "justification": "All chunk analyses failed"
        }
    
    # Weight scores by chunk size
    total_weight = 0
    weighted_sum = 0
    justifications = []
    
    for res in results:
        chunk_weight = len(res["justification"])  # Weight by justification length
        weighted_sum += res["score"] * chunk_weight
        total_weight += chunk_weight
        justifications.append(res["justification"])
    
    avg_score = weighted_sum / total_weight if total_weight > 0 else 5.0
    
    # Create combined justification
    if len(justifications) > 3:
        combined_justification = (
            f"Analyzed {len(results)} sections. Key findings: "
            f"{justifications[0]} {justifications[1]} Additional sections support."
        )
    else:
        combined_justification = " | ".join(justifications[:3])
    
    return {
        "score": round(avg_score, 1),
        "justification": combined_justification[:500]  # Limit length
    }

def determine_document_type(content: str, source: str, url: str) -> str:
    """Enhanced document type detection"""
    content_lower = content.lower()
    source_lower = source.lower()
    url_lower = url.lower()

    # Detection priorities
    if any(x in url_lower for x in ["ipcc.ch", "climatechange"]) or "ipcc" in source_lower:
        return "ipcc report"
    elif any(x in url_lower for x in ["arxiv.org", "biorxiv.org", "medrxiv.org"]):
        return "preprint"
    elif any(x in source_lower for x in ["nature", "science", "pnas", "thelancet", "cell"]):
        return "peer_reviewed"
    elif "journal" in source_lower or "review" in source_lower:
        return "peer_reviewed"
    else:
        return "study"

def analyze_study(content: str, url: str, title: str, authors: str, source: str) -> dict:
    """Main analysis function with enhanced reliability"""
    if len(content) < 1000:
        raise ValueError("Document content too short for analysis")
    
    doc_type = determine_document_type(content, source, url)
    processing_notes = [
        f"Document length: {len(content):,} characters",
        f"Detected document type: {doc_type}"
    ]
    
    analysis = {}
    all_scores = []
    
    for category, questions in empirical_questions.items():
        q_results = []
        cat_scores = []
        
        for question in questions:
            try:
                # Use chunking for large documents
                if len(content) > 10000:
                    result = ask_llm_with_chunks(content, question, doc_type)
                    processing_notes.append(f"Used chunk analysis for '{category}'")
                else:
                    result = ask_llm(content, question, doc_type)
                
                q_results.append({
                    "question": question,
                    "score": result["score"],
                    "justification": result["justification"]
                })
                cat_scores.append(result["score"])
                all_scores.append(result["score"])
                
            except Exception as e:
                logger.error(f"Question failed: {category}/{question} - {str(e)}")
                q_results.append({
                    "question": question,
                    "score": 5.0,
                    "justification": f"Analysis failed: {str(e)[:100]}"
                })
                processing_notes.append(f"Failed one question in {category}")
        
        # Calculate category average
        analysis[category] = {
            "questions": q_results,
            "subscore": round(sum(cat_scores)/len(cat_scores), 2) if cat_scores else 5.0
        }
    
    # Calculate overall score
    final_es_score = round(sum(all_scores) / len(all_scores), 2) if all_scores else 5.0
    
    # Enhanced classification
    empirical_score = analysis.get("empirical_basis", {}).get("subscore", 5.0)
    modeled_score = analysis.get("modeled_science", {}).get("subscore", 5.0)
    
    if empirical_score >= 7.5:
        classification = "Tier 1 (Empirical Science)"
        summary = "Strong empirical basis with direct measurements"
    elif modeled_score >= 7.5:
        classification = "Tier 2 (Modeled Science)"
        summary = "Relies primarily on computational models"
    else:
        classification = "Tier 3 (Scenario Analysis)"
        summary = "Focuses on narrative scenarios and projections"
    
    # Add quality indicators
    if len(content) < 3000:
        classification += " - Caution: Short Document"
        summary += " | Note: Limited content for full analysis"
    
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