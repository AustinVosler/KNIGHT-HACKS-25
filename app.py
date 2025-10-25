import webbrowser
from threading import Thread
from flask import Flask, jsonify, request, render_template, send_file
import os
import sqlite3
import shutil

FILE_PATH = 'videos/'
if os.path.exists(FILE_PATH):
    shutil.rmtree(FILE_PATH)
os.makedirs(FILE_PATH, exist_ok=True)

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

### ROUTES ###

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/record')
def record():
    return render_template('recorder.html')

@app.route('/gallery')
def gallery():
    return render_template('gallery.html')

### API ENDPOINTS ###

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
    
@app.route('/api/videos', methods=['POST'])
def add_video_route():
    data = request.files['file']
    path = os.path.join(FILE_PATH, data.filename)
    O_filename = data.filename
    video_id = add_video(path, O_filename)
    path = os.path.join(FILE_PATH, str(video_id) + ".webm")
    data.save(path)
    return jsonify({"id": video_id}), 201

@app.route('/api/videos/<int:id>', methods=['GET'])
def send_video(id):
    with database_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM videos WHERE id = ?', (id,))
        video = c.fetchone()
        if not video:
            return jsonify({"error": "Video not found"}), 404
    video_path = os.path.join(FILE_PATH, str(id) + ".webm")
    return send_file(video_path, mimetype='video/webm')

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == '__main__':
    conn = database_connection()
    initialize_db(conn)
    conn.close()

    # Thread(target=open_browser).start()
    app.run(debug=True)