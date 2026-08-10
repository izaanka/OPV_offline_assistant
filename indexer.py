import os
import sqlite3
import time
from typing import List

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "file_index.db")

def init_db():
    """Initialize the SQLite database and create the table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL UNIQUE,
            extension TEXT,
            last_modified REAL
        )
    ''')
    # Create an index on filename for fast LIKE queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename ON files(filename)')
    conn.commit()
    conn.close()

def build_index():
    """Scan common user directories and populate the database."""
    init_db()
    home = os.path.expanduser("~")
    search_dirs = [
        os.path.join(home, "Music"),
        os.path.join(home, "Downloads"),
        os.path.join(home, "Documents"),
        os.path.join(home, "Desktop"),
        os.path.join(home, "Videos"),
        os.path.join(home, "Pictures")
    ]
    
    file_records = []
    
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for root, dirs, files in os.walk(d):
            # Skip hidden directories and heavy dev directories
            dirs[:] = [d_name for d_name in dirs if not d_name.startswith('.') and d_name not in ('node_modules', 'venv', '__pycache__')]
            
            for f in files:
                if f.startswith('.'):
                    continue
                filepath = os.path.join(root, f)
                extension = os.path.splitext(f)[1].lower()
                try:
                    last_modified = os.path.getmtime(filepath)
                except OSError:
                    last_modified = 0.0
                
                file_records.append((f, filepath, extension, last_modified))

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Clear old index
    cursor.execute('DELETE FROM files')
    
    # Bulk insert new records
    cursor.executemany('''
        INSERT OR IGNORE INTO files (filename, filepath, extension, last_modified)
        VALUES (?, ?, ?, ?)
    ''', file_records)
    
    conn.commit()
    conn.close()
    
    return len(file_records)


import re
import difflib

def tokenize(text: str) -> set:
    """Normalize text and return set of word tokens."""
    base = os.path.splitext(text)[0]
    cleaned = re.sub(r'[^a-zA-Z0-9]', ' ', base.lower())
    tokens = set(cleaned.split())
    return {t for t in tokens if len(t) >= 2}


def query_index(query: str, limit: int = 10) -> List[str]:
    """Query the SQLite database for a filename matching the query."""
    if not os.path.exists(DB_PATH):
        build_index()
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    search_term = f'%{query}%'
    cursor.execute('''
        SELECT filepath FROM files 
        WHERE filename LIKE ? 
        LIMIT ?
    ''', (search_term, limit))
    
    results = [row[0] for row in cursor.fetchall()]
    conn.close()
    return results


VIDEO_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v', '.3gp'}
AUDIO_EXTS = {'.mp3', '.m4a', '.wav', '.flac', '.ogg', '.aac', '.opus', '.wma', '.alac'}
DOC_EXTS   = {'.txt', '.pdf', '.docx', '.doc', '.md', '.csv', '.xlsx', '.pptx', '.rtf', '.odt', '.json', '.py', '.log'}


def query_recent_files(category: str = "video", limit: int = 5) -> List[tuple]:
    """Query recent files by last_modified for a specific category ('video', 'audio', 'document')."""
    if not os.path.exists(DB_PATH):
        build_index()
        
    exts = set()
    cat_lower = category.lower()
    if any(w in cat_lower for w in ("video", "movie", "clip")):
        exts = VIDEO_EXTS
    elif any(w in cat_lower for w in ("audio", "music", "song", "track")):
        exts = AUDIO_EXTS
    elif any(w in cat_lower for w in ("doc", "text", "file", "pdf")):
        exts = DOC_EXTS
    else:
        exts = VIDEO_EXTS.union(AUDIO_EXTS).union(DOC_EXTS)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    placeholders = ','.join('?' for _ in exts)
    cursor.execute(f'''
        SELECT filepath, filename, last_modified FROM files
        WHERE extension IN ({placeholders})
        ORDER BY last_modified DESC
        LIMIT ?
    ''', list(exts) + [limit * 2])
    
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for fp, fname, mtime in rows:
        if os.path.exists(fp):
            results.append((fp, fname, 1.0))
            if len(results) >= limit:
                break
    return results


def query_index_smart(query: str, limit: int = 5) -> List[tuple]:
    """
    Search indexed files by computing relativistic similarity score on the filename.
    Supports recency queries like 'open most recent video file' or 'play latest music'.
    Returns list of (filepath, filename, score) tuples sorted by score descending.
    """
    q_lower = query.lower().strip()
    
    recency_keywords = ("recent", "latest", "newest", "last")
    if any(k in q_lower for k in recency_keywords):
        if any(w in q_lower for w in ("video", "movie", "clip")):
            return query_recent_files("video", limit=limit)
        elif any(w in q_lower for w in ("song", "music", "audio", "track")):
            return query_recent_files("audio", limit=limit)
        elif any(w in q_lower for w in ("doc", "document", "file", "text", "pdf")):
            return query_recent_files("document", limit=limit)
        else:
            return query_recent_files("all", limit=limit)

    if not os.path.exists(DB_PATH):
        build_index()
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT filename, filepath FROM files')
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return []
        
    query_tokens = tokenize(query)
    query_clean = re.sub(r'[^a-zA-Z0-9]', ' ', os.path.splitext(query)[0].lower()).strip()
    
    scored_files = []
    
    for filename, filepath in rows:
        if not os.path.exists(filepath):
            continue
            
        file_base_clean = re.sub(r'[^a-zA-Z0-9]', ' ', os.path.splitext(filename)[0].lower()).strip()
        file_tokens = tokenize(filename)
        
        if not file_tokens or not query_tokens:
            continue
            
        common_tokens = query_tokens.intersection(file_tokens)
        token_overlap_score = len(common_tokens) / len(query_tokens)
        
        seq_score = difflib.SequenceMatcher(None, query_clean, file_base_clean).ratio()
        
        containment_bonus = 0.0
        if query_clean and (query_clean in file_base_clean or file_base_clean in query_clean):
            containment_bonus = 0.3
            
        final_score = (0.5 * token_overlap_score) + (0.3 * seq_score) + containment_bonus
        
        if final_score > 0.15:
            scored_files.append((filepath, filename, final_score))
            
    scored_files.sort(key=lambda x: x[2], reverse=True)
    return [(fp, fname, score) for fp, fname, score in scored_files[:limit]]

if __name__ == "__main__":
    print("Building file index...")
    start_time = time.time()
    count = build_index()
    elapsed = time.time() - start_time
    print(f"Indexed {count} files in {elapsed:.2f} seconds.")
