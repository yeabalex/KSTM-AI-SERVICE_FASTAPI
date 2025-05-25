import hashlib

def get_history_hash(messages):
    return hashlib.md5(str([(msg.type, msg.content) for msg in messages]).encode()).hexdigest()