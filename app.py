from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"message": "This is your Flask backend!"})

if __name__ == '__main__':
    app.run(debug=True)