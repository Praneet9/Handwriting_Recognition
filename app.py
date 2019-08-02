from flask import Flask, render_template, jsonify, request
# from bson import json_util
import datetime
# import pytz
import math
from keras.models import model_from_json
import tensorflow as tf
import cv2
import numpy as np
import base64

global model, graph, label_dictionary

label_dictionary = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'a',
                    11: 'b', 12: 'd', 13: 'e', 14: 'f', 15: 'g', 16: 'h', 17: 'i', 18: 'j', 19: 'l', 20: 'm',
                    21: 'n', 22: 'q', 23: 'r', 24: 't', 25: 'y', 26: 'A', 27: 'B', 28: 'C', 29: 'D', 30: 'E',
                    31: 'F', 32: 'G', 33: 'H', 34: 'I', 35: 'J', 36: 'K', 37: 'L', 38: 'M', 39: 'N', 40: 'O',
                    41: 'P', 42: 'Q', 43: 'R', 44: 'S', 45: 'T', 46: 'U', 47: 'V', 48: 'W', 49: 'X', 50: 'Y',
                    51: 'Z'}

json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model/model.h5")

graph = tf.get_default_graph()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')




@app.route("/predict", methods = ['POST'])
def predict():
    test = request.get_data().decode()
    content = test.split(';')[1]
    image_encoded = content.split(',')[1]
    img = base64.decodebytes(image_encoded.encode('utf-8'))
    image_path = "test.jpg"
    with open(image_path, 'wb') as f:
        f.write(img)
    img = cv2.imread(image_path, 0)
    output = cv2.resize(img, (28, 28)).copy()
    output = cv2.threshold(output, 150, 255, cv2.THRESH_BINARY_INV)[1]
    output = output / 255
    with graph.as_default():
        proba = model.predict(output.reshape(-1, 28, 28, 1))
    print(proba)
    idx = np.argmax(proba)
    label = label_dictionary[idx] + " ====> " + str(np.max(proba) * 100)[:5] + "%"
    return label

if __name__ == '__main__':
    app.run(debug=True)