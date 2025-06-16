import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from datetime import datetime, timezone
from typing import List, Dict, Optional
import requests
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import fitz
from bs4 import BeautifulSoup
import json
import re

# Use ArxivLoader from langchain_community
from langchain_community.document_loaders import ArxivLoader

# Request and response models
class AnalyzeRequest(BaseModel):
    url: HttpUrl
    id:List[int]

class QuestionResult(BaseModel):
    question: str
    score: int
    justification: str

class CategoryResult(BaseModel):
    questions: List[QuestionResult]
    final_es_score: float
    summary: str

class AnalysisResult(BaseModel):
    empirical_basis: Optional[CategoryResult] = None
    modeled_science: Optional[CategoryResult] = None
    storyline_science: Optional[CategoryResult] = None
    final_es_score: Optional[float] = None
    classification: Optional[str] = None
    summary: Optional[str] = None

class AnalyzeResponse(BaseModel):
    title: str
    authors: str
    source: str
    url: HttpUrl
    analysis: AnalysisResult
    id: List[int]
    generated_at: datetime

ID_TO_CATEGORY = {
    1: "empirical_basis",
    2: "modeled_science",
    3: "storyline_science"
}

# FastAPI app and LLM initialization
app = FastAPI()

if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("Missing OPENAI_API_KEY in environment variables.")

oai_key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, openai_api_key=oai_key)

# Fixed questions grouped by category
CATEGORIES = {
    "empirical_basis": [
        "Does the study use direct measurements from instruments (e.g., thermometers, satellites, sensors)?",
        "Does the study rely on historical records (e.g., ship logs, written weather observations) as primary data?",
        "Does the analysis include statistically robust sampling of observations?",
        "Are uncertainties and error bars reported for empirical measurements?",
        "Is raw observational data made available or linked?"
    ],
    "modeled_science": [
        "Does the study use climate models to simulate scenarios?",
        "Are model parameters calibrated against observational datasets?",
        "Does the paper validate model outputs with independent data?",
        "Are multiple model runs/ensembles used to capture variability?",
        "Is sensitivity analysis on model inputs performed?"
    ],
    "storyline_science": [
        "Does the study present a narrative connecting observations and projections?",
        "Are causal mechanisms between drivers and outcomes clearly articulated?",
        "Does the paper discuss policy or real-world implications?",
        "Is there a comparison of alternative hypotheses or counterfactuals?",
        "Does the study weave empirical and modeled results into a cohesive story?"
    ]
}

BASE_SYSTEM_PROMPT = (
    "You are an AI assistant evaluating climate science papers using the Empirical Score framework. Your name is unit 007. "
    "For each question, provide a score from 1 to 10 and a brief (max 2 sentences) justification."
    "Return only valid JSON with a 'questions' array containing question, score, and justification."
)

def extract_text_and_metadata_from_url(url: str):
    # --- arXiv PDF or abstract page ---
    if "arxiv" in url:
        match = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)', url)
        arxiv_id = match.group(1) if match else None

        if arxiv_id:
            # Use ArxivLoader for robust metadata and content extraction
            try:
                loader = ArxivLoader(query=arxiv_id)
                docs = loader.load()
            except Exception as e:
                raise ValueError(f"Failed to load document from arXiv using ArxivLoader: {e}")
            if docs:
                doc = docs[0]
                text = doc.page_content
                meta = doc.metadata
                title = meta.get("Title") or meta.get("title") or "Unknown Title"
                authors = ", ".join(meta.get("Authors", [])) if meta.get("Authors") else meta.get("authors", "Unknown Authors")
                if isinstance(authors, list):
                    authors = ", ".join(authors)
                source = "arXiv"
                if len(text) < 500:
                    raise ValueError("The extracted arXiv content seems too short to be a scientific study.")
                return text, title, authors, source
            else:
                raise ValueError("Failed to load document from arXiv.")
        else:
            raise ValueError("Could not extract arXiv ID from URL.")

    # --- PDF extraction for non-arXiv PDFs ---
    if 'pdf' in url.lower():
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch the PDF URL: {e}")
        pdf_bytes = response.content
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        if len(text) < 500:
            raise ValueError("The extracted PDF content seems too short to be a scientific study.")
        title = "Unknown Title"
        authors = "Unknown Authors"
        source = url.split('/')[2]
        return text, title, authors, source

    # --- HTML extraction as before ---
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch the URL: {e}")
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(strip=True)
    if len(text) < 500:
        raise ValueError("The extracted content seems too short to be a scientific study.")

    # Title
    title = None
    if soup.find("meta", attrs={"name": "citation_title"}):
        title = soup.find("meta", attrs={"name": "citation_title"}).get("content")
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()
    else:
        title = "Unknown Title"

    # Authors
    authors = []
    for meta in soup.find_all("meta", attrs={"name": "citation_author"}):
        if meta.get("content"):
            authors.append(meta["content"].strip())
    if not authors:
        for meta in soup.find_all("meta", attrs={"name": "author"}):
            if meta.get("content"):
                authors.append(meta["content"].strip())
    if not authors:
        for meta in soup.find_all("meta", attrs={"property": "article:author"}):
            if meta.get("content"):
                authors.append(meta["content"].strip())
    authors_str = ", ".join(authors) if authors else "Unknown Authors"

    # Source (journal or publisher)
    source = None
    if soup.find("meta", attrs={"name": "citation_journal_title"}):
        source = soup.find("meta", attrs={"name": "citation_journal_title"}).get("content")
    elif soup.find("meta", attrs={"name": "citation_publisher"}):
        source = soup.find("meta", attrs={"name": "citation_publisher"}).get("content")
    else:
        source = url.split('/')[2]

    return text, title, authors_str, source

async def analyze_category(content: str, questions: List[str]) -> CategoryResult:
    # Truncate large texts for prompt
    snippet = content[:10000]
    prompt = (
        BASE_SYSTEM_PROMPT
        + "\nStudy Content (truncated):\n" + snippet
        + "\nQuestions: " + " | ".join(questions)
    )
    resp = llm([
        SystemMessage(content=BASE_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    raw = resp.content.strip()
    # Remove code block markers if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        # Remove the first line if it's ``` or ```json
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[0].strip().lower() == "json":
            lines = lines[1:]
        # Remove the last line if it's ```
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        result = json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse JSON: {e}\nLLM response: {resp.content}")

    scores = []
    q_results = []
    for q in result.get("questions", []):
        sc = int(q.get("score", 0))
        justification = q.get("justification", "")
        q_results.append(QuestionResult(question=q.get("question"), score=sc, justification=justification))
        scores.append(sc)

    sub = round(sum(scores) / len(scores), 2) if scores else 0.0
    return CategoryResult(
    questions=q_results,
    final_es_score=sub,
    summary="sucess!!"
    )

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    url = str(request.url)
    ids = request.id

    # Extract text and metadata
    try:
        text, title_txt, authors, source = extract_text_and_metadata_from_url(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text:
        raise HTTPException(status_code=400, detail="No text extracted.")

    analysis_data: Dict[str, CategoryResult] = {}
    all_scores: List[int] = []

    # Only analyze requested categories
    for i in ids:
        cat = ID_TO_CATEGORY.get(i)
        if not cat:
            continue
        qs = CATEGORIES[cat]
        cat_res = await analyze_category(text, qs)
        scores = [q.score for q in cat_res.questions]
        final_es_score = round(sum(scores) / len(scores), 2) if scores else 0.0
        summary = f"This study's relevance to {cat.replace('_', ' ')} is reflected in its Empirical Score of {final_es_score}."
        analysis_data[cat] = CategoryResult(
            questions=cat_res.questions,
            final_es_score=final_es_score,
            summary=summary
        )
        all_scores.extend(scores)

    # Build the AnalysisResult with only requested categories
    analysis_result_kwargs = {}
    for cat in analysis_data:
        analysis_result_kwargs[cat] = analysis_data[cat]

    # Add overall scores if more than one category is requested
    if all_scores:
        final = round(sum(all_scores) / len(all_scores), 2)
        if final >= 7:
            cls = "Tier 1 (Empirical Science)"
        elif final >= 4:
            cls = "Tier 2 (Modeled Science)"
        else:
            cls = "Tier 3 (Storyline Science)"
        summary = f"Overall ES score: {final}, classified as {cls.split('(')[1]}."
        analysis_result_kwargs["final_es_score"] = final
        analysis_result_kwargs["classification"] = cls
        analysis_result_kwargs["summary"] = summary
    
    result =AnalyzeResponse(
        title=title_txt,
        authors=authors,
        source=source,
        url=request.url,
        analysis=AnalysisResult(**analysis_result_kwargs),
        id=ids,
        generated_at=datetime.now(timezone.utc)
    )
    # Return the result with non-null fields only
    return result.model_dump(exclude_none=True)