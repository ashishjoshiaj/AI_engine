import requests
from bs4 import BeautifulSoup
import io
import fitz

def extract_text_and_metadata_from_url(url: str):
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
    # citation_author
    for meta in soup.find_all("meta", attrs={"name": "citation_author"}):
        if meta.get("content"):
            authors.append(meta["content"].strip())
    # Fallback 1
    if not authors:
        for meta in soup.find_all("meta", attrs={"name": "author"}):
            if meta.get("content"):
                authors.append(meta["content"].strip())
    # Fallback 2
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
        source = url.split('/')[2]  #domain

    return text, title, authors_str, source

def extract_text_and_metadata_from_pdf_url(url: str):
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch the PDF URL: {e}")

    try:
        pdf_bytes = io.BytesIO(response.content)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Failed to open PDF document: {e}")

    # Extract text from all pages
    text = ""
    for page in doc:
        text += page.get_text()

    if len(text) < 500:
        raise ValueError("The extracted PDF content seems too short to be a scientific study.")

    # Extract metadata
    metadata = doc.metadata
    title = metadata.get("title") or "Unknown Title"
    authors_str = metadata.get("author") or "Unknown Authors"
    source = url.split('/')[2]  # domain as fallback

    return text, title, authors_str, source