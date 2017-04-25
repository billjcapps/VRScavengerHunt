import Lab10Shared
import base64
import tensorflow as tf

from flask import Flask, jsonify, render_template, request
from flask_cors import cross_origin

print ("Beginng to launch REST service via FLASK")


app = Flask(__name__)


@app.route('/tensor/test', methods=['GET'])
@cross_origin()
def test():
    return jsonify(status='SUCCESS',message='I am up and running')


@app.route('/tensor/predict', methods=['POST'])
@cross_origin()
def predict():
    data = request.values['imageBase64']
    with open("webImages/predictImage.jpg", "wb") as fh:
        fh.write(base64.standard_b64decode(data))

    image_path = 'webImages/predictImage.jpg'

    result = Lab10Shared.predictor.predict_image(image_path)
    return jsonify(status='SUCCESS!', prediction=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
