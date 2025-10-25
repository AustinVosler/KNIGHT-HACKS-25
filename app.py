import webbrowser
from threading import Thread
from flask import Flask, jsonify, request, render_template
import os
import sqlite3

# Connect to SQLite database (or create it)
conn = sqlite3.connect('videos.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        url TEXT NOT NULL
    )
''')
conn.commit()

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/api/record')
def record():
    return render_template('recorder.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"message": "This is your Flask backend!"})

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == '__main__':
    Thread(target=open_browser).start()
    app.run(debug=True)