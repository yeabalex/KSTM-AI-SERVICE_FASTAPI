import hashlib
import logging
import pickle
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)

def create_cache_key(path):
    return hashlib.md5(path.encode()).hexdigest()

def read_txt_file(file_path):
    """
    Reads the content of a plain text file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: Entire text content.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_and_process_txt(
    file_path,
    chunk_size=1000,
    chunk_overlap=200,
    cache_dir=Path(".cache/txt"),
    refresh=False
):
    """
    Load and process a plain text file with optional caching and chunking.

    Args:
        file_path (str): Path to the text file.
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
                logging.info("Loaded text from cache.")
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"Cache load failed: {e}. Reloading.")

    try:
        logging.info(f"Reading text file: {file_path}")
        text = read_txt_file(file_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(text)

        documents = [Document(page_content=chunk, metadata={"source_file": str(file_path)}) for chunk in chunks]

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(documents, f)
                logging.info("Cached processed text successfully.")
        except Exception as e:
            logging.error(f"Failed to cache text: {e}")

        return documents

    except Exception as e:
        logging.error(f"Error processing text: {e}")
        return []
