import os
import pickle
import hashlib

CACHE_DIR= os.path.join(os.path.dirname(__file__), '.pdf_cache')

def get_cache_path(pdf_path):
    if not os.path.exists(pdf_path):
        return None
    
    mtime= os.path.getmtime(pdf_path)
    key= f"{pdf_path}:{mtime}"
    hash_name= hashlib.md5(key.encode()).hexdigest()+ '.pkl'

    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, hash_name)

def load_cache(pdf_path):
    cache_path= get_cache_path(pdf_path)
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError):
            return None
    return None

def save_cache(pdf_path, data):
    cache_path= get_cache_path(pdf_path)
    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

def clear_cache():
    if os.path.exists(CACHE_DIR):
        for f in os.listdir(CACHE_DIR):
            os.remove(os.path.join(CACHE_DIR, f))
        print("Cache cleared.")
    else:
        print("No cache to clear.")