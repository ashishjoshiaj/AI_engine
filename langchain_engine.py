import os
import json
from datetime import datetime
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from questions import empirical_questions
import re

if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("Missing OPENAI_API_KEY in environment variables.")

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3,streaming=True)

def chunk_text(text: str, chunk_size: int = 4000, chunk_overlap: int = 200) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(text)

def ask_llm(content, question, document_type="study"):
    if len(content) > 6000:
        return ask_llm_with_chunks(content, question, document_type)
    
    if "ipcc" in content.lower() or "intergovernmental panel" in content.lower():
        system_prompt = (
            "You are Unit 007, an AI model designed to evaluate the scientific rigor and validity of climate science studies. "
            "You are analyzing an institutional climate assessment. "
            "Your task is to analyze the provided document considering its nature as a comprehensive assessment. "
            "For the following question, return a JSON object with a 'score' (real number 1-10) and a 'justification' (1-2 sentences)."
        )
    elif "arxiv" in content.lower() or document_type == "preprint":
        system_prompt = (
            "You are Unit 007, an AI model designed to evaluate the scientific rigor and validity of climate science studies. "
            "You are analyzing a preprint or working paper that may not have undergone full peer review. "
            "Consider this context when evaluating the scientific rigor and validity. "
            "For the following question, return a JSON object with a 'score' (real number 1-10) and a 'justification' (1-2 sentences)."
        )
    else:
        system_prompt = (
            "You are Unit 007, an AI model designed to evaluate the scientific rigor and validity of climate science studies. "
            "Your task is to analyze the provided climate science study in its entirety. "
            "This includes a thorough review of the complete text, all data visualizations, figures, tables, and images presented within the document. "
            "For the following question, return a JSON object with a 'score' (real number 1-10) and a 'justification' (1-2 sentences)."
        )

    content_excerpt = content[:4000] if len(content) > 4000 else content
    
    user_prompt = f"""Study Content:
{content_excerpt}

Question: {question}

Return JSON:
{{
  "score": <real number between 1-10>,
  "justification": "<precise explanation of 1 or 2 sentences>"
}}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    try:
        response = llm(messages)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.lstrip("`")
            lines = raw.splitlines()
            if lines and lines[0].strip().lower() == "json":
                lines = lines[1:]
            if lines and lines[-1].strip() == "":
                lines = lines[:-1]
            raw = "\n".join(lines).strip("`").strip()
        
        parsed = json.loads(raw)
        return parsed
        
    except json.JSONDecodeError as e:
        try:
            score_match = re.search(r'"score":\s*([0-9.]+)', raw)
            justification_match = re.search(r'"justification":\s*"([^"]+)"', raw)
            
            if score_match and justification_match:
                return {
                    "score": float(score_match.group(1)),
                    "justification": justification_match.group(1)
                }
        except:
            pass
        
        raise ValueError(f"LLM response was not valid JSON: {e}\nResponse: {raw}")
    
    except Exception as e:
        raise ValueError(f"Error processing LLM response: {e}\nResponse: {response.content}")

def ask_llm_with_chunks(content, question, document_type="study"):
    
    chunks = chunk_text(content)
    chunk_scores = []
    chunk_justifications = []
    for i, chunk in enumerate(chunks[:5]):  # Limit to first 5 chunks for performance
        try:
            result = ask_llm(chunk, question, document_type)
            chunk_scores.append(result["score"])
            chunk_justifications.append(result["justification"])
        except Exception as e:
            print(f"Warning: Failed to process chunk {i+1}: {e}")
            continue
    
    if not chunk_scores:
        raise ValueError("Failed to process any chunks of the document")
    
    # Aggregate results
    avg_score = sum(chunk_scores) / len(chunk_scores)
    
    # Create combined justification
    combined_justification = f"Analysis of {len(chunk_scores)} document sections shows: {chunk_justifications[0]}"
    if len(chunk_justifications) > 1:
        combined_justification += f" Additional sections {('support' if avg_score > 5 else 'reinforce')} this assessment."
    
    return {
        "score": round(avg_score, 1),
        "justification": combined_justification
    }

def determine_document_type(content: str, source: str, url: str) -> str:
    content_lower = content.lower()
    source_lower = source.lower()
    url_lower = url.lower()
    
    if ("ipcc" in source_lower  or "intergovernmental panel" in content_lower or
        "climate change assessment" in content_lower):
        return "ipcc report"
    elif ("arxiv" in source_lower or "arxiv" in url_lower or
          "preprint" in content_lower):
        return "preprint"
    elif any(journal in source_lower for journal in 
             ["nature", "science", "journal", "proceedings"]):
        return "peer_reviewed"
    else:
        return "study"

def analyze_study(content: str, url: str, title: str, authors: str, source: str) -> dict:
    doc_type = determine_document_type(content, source, url)
    
    analysis = {}
    all_scores = []
    processing_notes = []
    content_length = len(content)
    processing_notes.append(f"Document length: {content_length:,} characters")
    
    if content_length > 10000:
        processing_notes.append("Large document processed in chunks")
    
    for category, questions in empirical_questions.items():
        q_results = []
        cat_scores = []
        
        for q in questions:
            try:
                if len(q) < 6000:
                    result = ask_llm_with_chunks(content, q, doc_type)
                else:
                    result = ask_llm(content, q, doc_type)
                q_results.append({
                    "question": q,
                    "score": result["score"],
                    "justification": result["justification"]
                })
                cat_scores.append(result["score"])
                all_scores.append(result["score"])
                
            except Exception as e:
                print(f"Warning: Failed to process question '{q}': {e}")
                processing_notes.append(f"Failed to process 1 question in {category}")
                continue
        
        if cat_scores:
            subscore = round(sum(cat_scores) / len(cat_scores), 2)
            analysis[category] = {
                "questions": q_results,
                "subscore": subscore
            }
        else:
            # Fallback for failed categories
            analysis[category] = {
                "questions": [],
                "subscore": 4.0  # Neutral score
            }
            processing_notes.append(f"Category {category} failed to process")

    if not all_scores:
        raise ValueError("Failed to process any questions for this document")

    final_es_score = round(sum(all_scores) / len(all_scores), 2)

    # Enhanced classification logic based on document type
    empirical_score = analysis.get("empirical_basis", {}).get("subscore", 5.0)
    modeled_score = analysis.get("modeled_science", {}).get("subscore", 5.0)
    storyline_score = analysis.get("storyline_science", {}).get("subscore", 5.0)

    if empirical_score >= 7:
        classification = "Tier 1 (Empirical Science)"
        summary = (
            "This study is primarily empirical, relying on direct observations, "
            "measurements, and data-driven analysis."
        )
    elif modeled_score >= 7:
        classification = "Tier 2 (Modeled Science)"
        summary = (
            "This study primarily uses computational models and statistical methods "
            "to understand climate processes and make projections."
        )
    else:
        classification = "Tier 3 (Storyline Science)"
        summary = (
            "This study primarily develops narratives and scenarios, with limited "
            "direct empirical validation."
        )

    return {
        "title": title,
        "authors": authors,
        "source": source,
        "url": url,
        "document_type": doc_type,
        "analysis": {
            "empirical_basis": analysis.get("empirical_basis", {"questions": [], "subscore": 4.0}),
            "modeled_science": analysis.get("modeled_science", {"questions": [], "subscore": 4.0}),
            "storyline_science": analysis.get("storyline_science", {"questions": [], "subscore": 4.0}),
            "final_es_score": final_es_score,
            "classification": classification,
            "summary": summary
        },
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z"
    }