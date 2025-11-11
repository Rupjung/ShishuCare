import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="infant_cry",
        user="postgres",
        password="2060"
    )

def insert_result(user, label, confidence):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO results (username, prediction, confidence) VALUES (%s, %s, %s)", (user, label, confidence))
    conn.commit()
    conn.close()
