import json
from flask import Flask, jsonify, request
app = Flask(__name__)
@app.route('/', methods=['GET'])

def query():
    image1 = request.args.get('image1')
    image2 = request.args.get('image2')
    image3 = request.args.get('image3')
    image4 = request.args.get('image4')
    image5 = request.args.get('image5')

    return(image1 + " " + image2)
app.run()