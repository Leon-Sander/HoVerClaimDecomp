import sqlite3

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