import sqlite3

conn = sqlite3.connect('cache.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_uuid TEXT
    )
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_uuid TEXT,
        vector TEXT
    )
''')
conn.commit()
conn.close()
