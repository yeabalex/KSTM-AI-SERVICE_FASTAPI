import json
import hashlib
import logging
import pickle
import requests
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)

def create_cache_key(path):
    return hashlib.md5(path.encode()).hexdigest()

def flatten_json(obj, parent_key='', sep='.'):
    items = {}
    for k, v in obj.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def read_json_from_local(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_json_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Failed to fetch JSON from {url}, status code: {response.status_code}")
        return []

def read_json(file_path):
    if file_path.startswith("http://") or file_path.startswith("https://"):
        data = read_json_from_url(file_path)
    else:
        data = read_json_from_local(file_path)

    if isinstance(data, dict):
        data = [data]

    flat_rows = []
    for entry in data:
        flat = flatten_json(entry)
        row_str = ', '.join(f"{k}: {v}" for k, v in flat.items())
        flat_rows.append(row_str)

    return "\n".join(flat_rows)

def load_and_process_json(
    file_path,
    chunk_size=1000,
    chunk_overlap=200,
    cache_dir=Path(".cache/json"),
    refresh=False
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = create_cache_key(file_path)
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
        full_text = read_json(file_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(full_text)

        documents = [Document(page_content=chunk, metadata={"source_file": file_path}) for chunk in chunks]

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
