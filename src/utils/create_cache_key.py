import hashlib
import json


def create_cache_key(documents):
    """
    Create a unique, hashable key for caching based on document content
    """
    # Extract unique identifiable information from documents
    doc_summary = [
        {
            'page_content': doc.page_content[:100],  # First 100 chars
            'source': doc.metadata.get('source', '')
        } for doc in documents
    ]
    
    # Convert to a JSON string
    summary_json = json.dumps(doc_summary, sort_keys=True)
    
    # Create a hash
    return hashlib.md5(summary_json.encode()).hexdigest()