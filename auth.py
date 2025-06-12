from db_config import fetch_query, execute_query
import hashlib

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, role):
    hashed_pw = hash_password(password)
    query = "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)"
    execute_query(query, (username, hashed_pw, role))

def authenticate_user(username, password):
    hashed_pw = hash_password(password)
    query = "SELECT * FROM users WHERE username=%s AND password=%s"
    user = fetch_query(query, (username, hashed_pw))
    if user:
        return user[0]  # Return user dict
    return None
