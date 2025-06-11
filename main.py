from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from utils import extract_text_and_metadata_from_url, extract_text_and_metadata_from_pdf_url
from lancgchain_engine import analyze_study

app = FastAPI()

class AnalyzeRequest(BaseModel):
    url: HttpUrl  # Automatically validates format

@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    try:
        content, title, authors, source = extract_text_and_metadata_from_url(str(request.url))
        result = analyze_study(content, str(request.url),title, authors, source)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AnalyzePDFRequest(BaseModel):
    url: HttpUrl  # Automatically validates format

@app.post("/analyze/pdf")
async def analyze_pdf(request: AnalyzePDFRequest):
    try:
        content, title, authors, source = extract_text_and_metadata_from_pdf_url(str(request.url))
        result = analyze_study(content, str(request.url), title, authors, source)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
