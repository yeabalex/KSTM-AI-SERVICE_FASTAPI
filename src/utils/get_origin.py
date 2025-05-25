from urllib.parse import urlparse

def get_origin(url):
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"