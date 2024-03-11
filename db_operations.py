import sqlite3
import unicodedata

def connect_to_db(db_path = '../data/wiki_wo_links.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    return conn, cursor


#conn, cursor = connect_to_db(db_path)
#cursor.execute("SELECT * FROM documents WHERE id='Ride Like the Wind'")

def get_documents_batch(cursor, offset: int, batch_size: int) -> list[tuple]:
    query = f"SELECT * FROM documents LIMIT {batch_size} OFFSET {offset}"
    cursor.execute(query)
    results = cursor.fetchall()
    return results

def get_total_document_count(cursor):
    cursor.execute("SELECT COUNT(*) FROM documents")
    return cursor.fetchone()[0]

def get_doc_by_title(cursor, title):
    cursor.execute("SELECT * FROM documents WHERE id=?", (title,))
    output = cursor.fetchone()
    return output

def get_db_column_names(cursor):
    cursor.execute("PRAGMA table_info('documents');")
    columns = cursor.fetchall()
    return columns

def get_text_from_doc(cursor, title):
    doc = cursor.execute("SELECT text FROM documents WHERE id = ?",
                   (unicodedata.normalize("NFD", title),)).fetchall()[0][0]
    return doc