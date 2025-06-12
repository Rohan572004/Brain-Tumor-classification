from db_config import get_connection

def initialize_tables():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS doctors (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100),
        email VARCHAR(100) UNIQUE,
        password VARCHAR(100)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100),
        email VARCHAR(100) UNIQUE,
        dob DATE
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS scans (
        id SERIAL PRIMARY KEY,
        patient_id INT REFERENCES patients(id),
        doctor_id INT REFERENCES doctors(id),
        prediction VARCHAR(100),
        model_used VARCHAR(100),
        scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        image_path TEXT,
        gradcam_path TEXT
    );
    """)

    conn.commit()
    cur.close()
    conn.close()
