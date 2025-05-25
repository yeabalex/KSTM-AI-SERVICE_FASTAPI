import functools
import hashlib
import logging
import pickle
import time
from pathlib import Path

import bs4
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)

def create_cache_key(url):
    """Generate a unique cache key from the URL."""
    return hashlib.md5(url.encode()).hexdigest()

def extract_text_with_links(html_content):
    """
    Extract structured text including headings, paragraphs, lists, and links.

    Args:
        html_content (str): The raw HTML content.

    Returns:
        str: Extracted text with links preserved.
    """
    soup = bs4.BeautifulSoup(html_content, "html.parser")
    extracted_text = []

    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "a", 'div', 'article']):
        if element.name.startswith("h"):  # Headings
            extracted_text.append(f"\n{element.get_text(strip=True)}\n" + "=" * len(element.get_text(strip=True)))
        elif element.name == "p":  # Paragraphs
            extracted_text.append(element.get_text(strip=True))
        elif element.name == "li":  # List items
            extracted_text.append(f"- {element.get_text(strip=True)}")
        elif element.name == "a" and element.get("href"):  # Links
            extracted_text.append(f"[{element.get_text(strip=True)}]({element['href']})")  # Markdown-style link

    return "\n\n".join(extracted_text)

def fetch_dynamic_page_content(url):
    """
    Uses Selenium to load JavaScript-rendered pages and return HTML content.

    Args:
        url (str): The URL to fetch.

    Returns:
        str: Fully rendered HTML content.
    """
    options = Options()
    options.add_argument("--headless")  # Run in headless mode (no UI)
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # Setup WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        logging.info(f"Fetching page with Selenium: {url}")
        driver.get(url)
        time.sleep(5)  # Allow time for JavaScript to execute

        # Extract full page HTML
        html_content = driver.page_source
        return html_content
    finally:
        driver.quit()

@functools.lru_cache(maxsize=32)
def load_and_process_documents(
    url, 
    chunk_size=1000, 
    chunk_overlap=200, 
    cache_dir=Path('.cache/documents'),
    refresh=False
):
    """
    Load and process web documents with caching, structured extraction, and link preservation.

    Args:
        url (str): The URL to load and process.
        chunk_size (int, optional): Size of text chunks. Defaults to 1000.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 200.
        cache_dir (Path, optional): Directory to cache processed documents.
        refresh (bool, optional): Force refresh of cached document. Defaults to False.

    Returns:
        list: Processed document chunks (structured text with links).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_key = create_cache_key(url)
    cache_path = cache_dir / f"{cache_key}.pkl"
    
    # Check cache
    if not refresh and cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                logging.info("Loaded from cache.")
                return cached_data
        except (pickle.PickleError, EOFError) as e:
            logging.warning(f"Cache load failed: {e}. Reloading.")

    try:
        logging.info(f"Fetching data from: {url}")
        
        # Use Selenium to fetch JavaScript-rendered content
        html_content = fetch_dynamic_page_content(url)

        if not html_content:
            logging.error("Failed to load page content.")
            return []

        logging.info("Extracting structured content with links...")
        
        # Extract structured content
        structured_text = extract_text_with_links(html_content)

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        processed_documents = text_splitter.split_text(structured_text)

        if not processed_documents:
            logging.error("Document splitting returned an empty list.")
            return []

        logging.info(f"Split into {len(processed_documents)} chunks.")

        # Convert text chunks into Document objects with metadata
        doc_objects = [Document(page_content=chunk, metadata={"source_url": url}) for chunk in processed_documents]

        # Cache the documents
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(doc_objects, f)
            logging.info("Cached processed documents successfully.")
        except Exception as cache_error:
            logging.error(f"Failed to cache data: {cache_error}")

        return doc_objects  # Return Document objects instead of raw strings

    except Exception as e:
        logging.error(f"Failed to load/process document from {url}: {e}")
        return []
