import webbrowser
from threading import Thread
from flask import Flask, jsonify, request, render_template, send_file
import os
import sqlite3
import shutil
from moviepy import VideoFileClip
import uuid

temp_PATH = 'FILES/temp'
video_PATH = 'FILES/videos'
if os.path.exists(temp_PATH):
    shutil.rmtree(temp_PATH)
os.makedirs(temp_PATH, exist_ok=True)
os.makedirs(video_PATH, exist_ok=True)

FILE_PATH = 'FILES'

# Connect to SQLite database (or create it)
def database_connection():
    return sqlite3.connect('videos.db')

def initialize_db(conn):
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS unprocessed_videos')
    c.execute('''
        CREATE TABLE IF NOT EXISTS unprocessed_videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder TEXT NOT NULL,
            filename TEXT NOT NULL,
            O_filename TEXT NOT NULL,
            pair_id INTEGER UNIQUE DEFAULT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS processed_videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder TEXT NOT NULL,
            filename TEXT NOT NULL,
            O_filename TEXT NOT NULL,
            pair_id INTEGER UNIQUE
        )
    ''')
    conn.commit()

def add_video(folder, filename, O_filename):
    with database_connection() as conn: 
        c = conn.cursor()
        c.execute('INSERT INTO unprocessed_videos (folder, filename, O_filename) VALUES (?, ?, ?)', (folder, filename, O_filename))
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

# @app.route('/about')
# def about():
#     return render_template('about.html')

@app.route('/trimming')
def trimming():
    videos = [f for f in os.listdir(FILE_PATH) if f.endswith('.webm')]
    return render_template('trimming.html')

### API ENDPOINTS ###

@app.route('/api/videos', methods=['GET'])
def list_videos():
    with database_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM processed_videos')
        videos = c.fetchall()
        return jsonify(videos)
    
@app.route('/api/videos', methods=['POST'])
def add_video_route():
    data = request.files['file']
    newName = uuid.uuid4().hex + ".webm"
    O_filename = data.filename
    video_id = add_video("temp", newName, O_filename)
    path = os.path.join(FILE_PATH, "temp", newName)
    data.save(path)
    print(video_id)
    return jsonify({"id": video_id}), 201

@app.route('/api/videos/<int:id>', methods=['GET'])
def send_video(id):
    with database_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM processed_videos WHERE id = ?', (id,))
        video = c.fetchone()
        if not video:
            return jsonify({"error": "Video not found"}), 404
    path = os.path.join(FILE_PATH, "videos", video.filename)
    return send_file(path, mimetype='video/mp4')

@app.route('/api/videos/<int:id>', methods=['PUT'])
def save_video(id):
    with database_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM unprocessed_videos WHERE id = ?', (id,))
        video = c.fetchone()
        if not video:
            return jsonify({"error": "Video not found"}), 404

        temp_path = os.path.join(FILE_PATH, "temp", video[2])  # filename
        final_path = os.path.join(FILE_PATH, "videos", video[2])

        # Ensure videos folder exists
        os.makedirs(os.path.join(FILE_PATH, "videos"), exist_ok=True)

        if not os.path.exists(temp_path):
            return jsonify({"error": "Temp video not found"}), 404

        shutil.move(temp_path, final_path)
        c.execute('UPDATE videos SET folder = ? WHERE id = ?', ("videos", id))
        conn.commit()

    return jsonify({"message": "Video moved to videos folder"}), 200

@app.route('/api/videos/<int:id>', methods=['DELETE'])
def delete_video(id):
    with database_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM processed_videos WHERE id = ?', (id,))
        video = c.fetchone()
        c.execute('DELETE FROM processed_videos WHERE id = ?', (id,))
        conn.commit()
    path = os.path.join(FILE_PATH, "videos", video.filename)
    if os.path.exists(path):
        os.remove(path)
    return jsonify({"message": "Video deleted"}), 204

@app.route('/api/videos/process/<int:id>', methods=['PUT'])
def process_video(id):
    with database_connection() as conn:
        c = conn.cursor()
        # Find the unprocessed video
        c.execute('SELECT * FROM unprocessed_videos WHERE id = ?', (id,))
        unprocessed = c.fetchone()
        if not unprocessed:
            return jsonify({"error": "Unprocessed video not found"}), 404

        # Build path to the unprocessed video
        input_path = os.path.join(unprocessed[1], unprocessed[2])  # folder, filename

        # --- Process the video here ---
        # processed_video_path = FUNCTION
        processed_video_path = "temp"
        processed_filename = uuid.uuid4().hex + ".webm"
        
        processed_video_path.rename(FILE_PATH, "videos", processed_filename)
        

        # Add processed video to DB
        c.execute('INSERT INTO processed_videos (folder, filename, O_filename, pair_id) VALUES (?, ?, ?, ?)',
                  ("videos", processed_filename, unprocessed[3], id))
        processed_id = c.lastrowid

        # Update unprocessed video with pair_id
        c.execute('UPDATE unprocessed_videos SET pair_id = ? WHERE id = ?', (processed_id, id))
        conn.commit()

    return jsonify({"message": "Video processed and linked", "processed_id": processed_id}), 201

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == '__main__':
    conn = database_connection()
    initialize_db(conn)
    conn.close()

    # Thread(target=open_browser).start()
    app.run(debug=True)