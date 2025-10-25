import webbrowser
from threading import Thread
from flask import Flask, jsonify, request, render_template
import os
import sqlite3

# Connect to SQLite database (or create it)
def database_connection():
    return sqlite3.connect('videos.db')

def initialize_db(conn):
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS videos')
    c.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            O_filename TEXT NOT NULL,
            processed BOOLEAN DEFAULT 0
        )
    ''')
    conn.commit()

def add_video(path, O_filename):
    with database_connection() as conn: 
        c = conn.cursor()
        c.execute('INSERT INTO videos (path, O_filename, processed) VALUES (?, ?, ?)', (path, O_filename, False))
        conn.commit()
        return c.lastrowid

app = Flask(__name__)
#baka

@app.route('/index')
def home():
    return "Hello, Flask!"

@app.route('/api/record')
def record():
    return render_template('recorder.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"message": "This is your Flask backend!"})

@app.route('/api/videos', methods=['GET'])
def list_videos():
    with database_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM videos')
        videos = c.fetchall()
        return jsonify(videos)

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == '__main__':
    conn = database_connection()
    initialize_db(conn)
    conn.close()

    add_video('/path/to/video1.mp4', 'output1.mp4')
    add_video('/path/to/video2.mp4', 'output2.mp4')

    Thread(target=open_browser).start()
    app.run(debug=True)