"""Monitoring utilities:
- Logs queries, retrieved sources, answers, latency, timestamp, and quality (if provided)
- Saves to CSV and SQLite for analysis
"""
import csv
import sqlite3
import time
from datetime import datetime
import json
from config import CSV_LOG_PATH, SQLITE_DB

CSV_FIELDS = ["timestamp", "query", "answer", "sources_json", "latency", "quality"]

def init_csv():
    if not CSV_LOG_PATH.exists():
        with open(CSV_LOG_PATH, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()

def log_to_csv(query, answer, sources, latency, quality=None):
    init_csv()
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "answer": answer,
        "sources_json": json.dumps(sources),
        "latency": float(latency),
        "quality": quality or ""
    }
    with open(CSV_LOG_PATH, "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)

def init_sqlite():
    conn = sqlite3.connect(SQLITE_DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            answer TEXT,
            sources_json TEXT,
            latency REAL,
            quality TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_to_sqlite(query, answer, sources, latency, quality=None):
    init_sqlite()
    conn = sqlite3.connect(SQLITE_DB)
    cur = conn.cursor()
    cur.execute("INSERT INTO queries (timestamp, query, answer, sources_json, latency, quality) VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), query, answer, json.dumps(sources), float(latency), quality or ""))
    conn.commit()
    conn.close()

def log(query, answer, sources, latency, quality=None):
    try:
        log_to_csv(query, answer, sources, latency, quality=quality)
    except Exception as e:
        print("Failed to write CSV log:", e)
    try:
        log_to_sqlite(query, answer, sources, latency, quality=quality)
    except Exception as e:
        print("Failed to write SQLite log:", e)
