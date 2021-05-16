from flask import Flask, jsonify, request
from ml import final_file
from PIL import Image, ImageFile
from io import BytesIO
import base64
import numpy
ImageFile.LOAD_TRUNCATED_IMAGES = True
app = Flask(__name__)
@app.route('/', methods=['GET'])

def query():
    final_model = final_file.get_model("ml/hackethernet.h5")
    image = request.args.get('image')
    base64Data = image + "=="
    img = Image.open(BytesIO(base64.b64decode(base64Data)))
    imgArray = numpy.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
    print(imgArray)
    print(imgArray.size)
    response = final_model.predict(imgArray)
    return jsonify (
        _class = str(response[0]),
        _prob = float(response[1]),
    )
app.run()
