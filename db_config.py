import psycopg2
from psycopg2.extras import RealDictCursor
def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="brain_tumor_db",
        user="postgres",
        password="rohankanthale@2004",
        port="5432"  # default PostgreSQL port
    )

def fetch_query(query, params=None):
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(query, params)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result

def execute_query(query, params=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    cursor.close()
    conn.close()