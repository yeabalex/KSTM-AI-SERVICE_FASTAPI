import csv
import hashlib
import logging
import pickle
import requests
import io
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)

def create_cache_key(path, delimiter):
    key = f"{path}_{delimiter}"
    return hashlib.md5(key.encode()).hexdigest()

def read_csv_file(file_path, delimiter=','):
    """
    Reads a CSV file from a local path or URL and returns a list of dictionaries (rows).

    Args:
        file_path (str): Path to the CSV file or URL.
        delimiter (str): Delimiter used in the CSV file.

    Returns:
        list: List of row strings.
    """
    # Check if file_path is a URL or a local path
    if file_path.startswith('http://') or file_path.startswith('https://'):
        return read_csv_from_url(file_path, delimiter)
    else:
        return read_csv_from_local(file_path, delimiter)

def read_csv_from_url(url, delimiter=','):
    """
    Reads a CSV file from a URL and returns a list of dictionaries (rows).

    Args:
        url (str): URL to the CSV file.
        delimiter (str): Delimiter used in the CSV file.

    Returns:
        list: List of row strings.
    """
    response = requests.get(url)
    if response.status_code == 200:
        csvfile = io.StringIO(response.text)
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        rows = []
        for row in reader:
            row_str = ', '.join(f"{k}: {v}" for k, v in row.items())
            rows.append(row_str)
        return rows
    else:
        logging.error(f"Failed to fetch CSV from {url}, status code: {response.status_code}")
        return []

def read_csv_from_local(file_path, delimiter=','):
    """
    Reads a CSV file from a local path and returns a list of dictionaries (rows).

    Args:
        file_path (str): Path to the local CSV file.
        delimiter (str): Delimiter used in the CSV file.

    Returns:
        list: List of row strings.
    """
    rows = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in reader:
            row_str = ', '.join(f"{k}: {v}" for k, v in row.items())
            rows.append(row_str)
    return rows

def load_and_process_csv(
    file_path, 
    delimiter=',', 
    chunk_size=1000, 
    chunk_overlap=200, 
    cache_dir=Path('.cache/csv'),
    refresh=False
):
    """
    Load and process a CSV file with optional caching and text chunking.

    Args:
        file_path (str): Path to the CSV file or URL.
        delimiter (str): Delimiter used in the CSV file.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap between chunks.
        cache_dir (Path): Directory to store cached files.
        refresh (bool): Force refresh of cache.

    Returns:
        list: LangChain Document objects.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create cache key from the path and delimiter
    cache_key = create_cache_key(str(file_path), delimiter)
    cache_path = cache_dir / f"{cache_key}.pkl"

    # Check for cached documents
    if not refresh and cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                logging.info("Loaded CSV from cache.")
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"Cache load failed: {e}. Reloading.")

    try:
        logging.info(f"Reading CSV file: {file_path}")
        row_texts = read_csv_file(file_path, delimiter)
        full_text = "\n".join(row_texts)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(full_text)

        documents = [Document(page_content=chunk, metadata={"source_file": str(file_path)}) for chunk in chunks]

        # Cache the processed documents
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(documents, f)
                logging.info("Cached processed CSV successfully.")
        except Exception as e:
            logging.error(f"Failed to cache CSV: {e}")

        return documents

    except Exception as e:
        logging.error(f"Error processing CSV: {e}")
        return []