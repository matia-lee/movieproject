from flask import Flask, request, jsonify
from main import main

app = Flask(__name__)

# Flask routes and logic go here

if __name__ == '__main__':
    app.run(debug=True)