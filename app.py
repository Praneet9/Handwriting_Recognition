from flask import Flask, render_template, jsonify, request
from bson import json_util
import datetime
import pytz
import math
from keras.models import model_from_json
import tensorflow as tf
import cv2
import numpy as np
import base64

global model, graph, label_dictionary

label_dictionary = {0: 'Early Blight', 1: 'Healthy', 2: 'Late Blight'}

# json_file = open('model/model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# model.load_weights("model/model.h5")

# graph = tf.get_default_graph()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/image", methods = ['POST'])
def predict():
    test = request.data.decode()
    # print(eval(test)['imageData'])
    # test.save('test.jpg')
    img = base64.b64decode(eval(test)['imageData'])
    image_path = "test.jpg"
    with open('image.jpg', 'wb') as f:
        f.write(img)
    # img = cv2.imread(image_path)
    # output = cv2.resize(img, (256, 256)).copy()
    # img = cv2.resize(img, (128, 128))
    # img = img / 255
    # with graph.as_default():
    #     proba = model.predict(img.reshape(-1, 128, 128, 3))

    # idx = np.argmax(proba)
    # label = '<h1>' + label_dictionary[idx] + " ====> " + str(np.max(proba) * 100)[:5] + "%" + '</h1>'
    return "label"

if __name__ == '__main__':
    app.run(debug=True)