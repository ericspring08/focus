from ml import final_file
from flask import Flask, jsonify, request
from PIL import Image, ImageFile
from io import BytesIO
import base64
import numpy
import re

ImageFile.LOAD_TRUNCATED_IMAGES = True
app = Flask(__name__)

@app.route('/go', methods=['POST'])
def go():
    final_model = final_file.get_model("ml/hackethernet.h5")
    image = request.get_json()['img']
    base64Data = image + "=="
    image_data = re.sub('^data:image/.+;base64,', '', base64Data)
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    rgb_im = img.convert('RGB')
    rgb_im.save('img.jpg')
    imgArray = numpy.array(rgb_im.getdata()).reshape(rgb_im.size[0], rgb_im.size[1], 3)
    response = final_model.predict(imgArray)
    return jsonify (
        _class = str(response[0]),
        _prob = float(response[1]),
    )

app.run()
