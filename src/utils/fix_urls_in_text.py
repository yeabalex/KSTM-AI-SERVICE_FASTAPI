import re

def fix_urls_in_text(text, base_url):
    base_url = base_url.rstrip('/')

    urls = re.findall(r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
   
    for url in urls:
        malformed_pattern = base_url + "https"
        if malformed_pattern in url:
            fixed_url = url.replace(malformed_pattern, base_url)
            text = text.replace(url, fixed_url)
   
    return text