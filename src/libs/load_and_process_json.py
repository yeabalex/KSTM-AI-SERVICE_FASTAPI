import json
import hashlib
import logging
import pickle
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)

def create_cache_key(path):
    return hashlib.md5(path.encode()).hexdigest()

def flatten_json(obj, parent_key='', sep='.'):
    """
    Flattens nested dictionaries into a single-level dict with dotted keys.

    Args:
        obj (dict): The JSON object.
        parent_key (str): The base key prefix for recursion.
        sep (str): Separator for nested keys.

    Returns:
        dict: Flattened key-value pairs.
    """
    items = {}
    for k, v in obj.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def read_json_file(file_path):
    """
    Loads JSON data and flattens it into readable strings.

    Args:
        file_path (str): Path to JSON file.

    Returns:
        list: List of row strings.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]  # Wrap single dict for uniformity

    flat_rows = []
    for entry in data:
        flat = flatten_json(entry)
        row_str = ', '.join(f"{k}: {v}" for k, v in flat.items())
        flat_rows.append(row_str)

    return flat_rows

def load_and_process_json(
    file_path,
    chunk_size=1000,
    chunk_overlap=200,
    cache_dir=Path(".cache/json"),
    refresh=False
):
    """
    Load and process a JSON file with optional caching and text chunking.

    Args:
        file_path (str): Path to the JSON file.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap between chunks.
        cache_dir (Path): Directory to store cached files.
        refresh (bool): Force refresh of cache.

    Returns:
        list: LangChain Document objects.
    """
    file_path = Path(file_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = create_cache_key(str(file_path))
    cache_path = cache_dir / f"{cache_key}.pkl"

    if not refresh and cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                logging.info("Loaded JSON from cache.")
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"Cache load failed: {e}. Reloading.")

    try:
        logging.info(f"Reading JSON file: {file_path}")
        row_texts = read_json_file(file_path)
        full_text = "\n".join(row_texts)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(full_text)

        documents = [Document(page_content=chunk, metadata={"source_file": str(file_path)}) for chunk in chunks]

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(documents, f)
                logging.info("Cached processed JSON successfully.")
        except Exception as e:
            logging.error(f"Failed to cache JSON: {e}")

        return documents

    except Exception as e:
        logging.error(f"Error processing JSON: {e}")
        return []
