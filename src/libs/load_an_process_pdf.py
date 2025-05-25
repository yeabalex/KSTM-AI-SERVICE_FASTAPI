import hashlib
import logging
import pickle
from pathlib import Path
from io import BytesIO

import requests
import fitz  # PyMuPDF

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)


def create_cache_key(source):
    return hashlib.md5(source.encode()).hexdigest()


def download_pdf_content(url):
    """Download the PDF file from the URL and return binary content."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        logging.info(f"Downloaded PDF from: {url}")
        return response.content
    except Exception as e:
        logging.error(f"Failed to download PDF from URL: {e}")
        return None


def read_local_pdf(file_path):
    """Read PDF content from a local file."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        logging.info(f"Read PDF from local file: {file_path}")
        return content
    except Exception as e:
        logging.error(f"Failed to read local PDF file: {e}")
        return None


def extract_text_from_pdf(pdf_bytes):
    """Extract text from the PDF using PyMuPDF."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {e}")
        return ""


def load_and_process_pdf(
    source,  # can be URL or local file path
    chunk_size=1000,
    chunk_overlap=200,
    cache_dir=Path('.cache/pdfs'),
    refresh=False,
    is_local_file=False
):
    """
    Load and process a PDF from a URL or local file.

    :param source: URL or local file path
    :param chunk_size: Size of each text chunk
    :param chunk_overlap: Overlap between text chunks
    :param cache_dir: Directory to store cache files
    :param refresh: If True, ignore cache and reprocess
    :param is_local_file: If True, treat 'source' as local file path
    :return: List of Document objects
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = create_cache_key(source)
    cache_path = cache_dir / f"{cache_key}.pkl"

    if not refresh and cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                logging.info("Loaded from cache.")
                return cached_data
        except (pickle.PickleError, EOFError) as e:
            logging.warning(f"Cache load failed: {e}. Reloading.")

    if is_local_file:
        pdf_bytes = read_local_pdf(source)
    else:
        pdf_bytes = download_pdf_content(source)

    if not pdf_bytes:
        return []

    text = extract_text_from_pdf(pdf_bytes)
    if not text.strip():
        logging.warning("Extracted text is empty.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    doc_objects = [Document(page_content=chunk, metadata={"source": source}) for chunk in chunks]

    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(doc_objects, f)
        logging.info("Cached processed PDF successfully.")
    except Exception as cache_error:
        logging.error(f"Failed to cache data: {cache_error}")

    return doc_objects
