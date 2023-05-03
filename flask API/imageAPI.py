
import numpy as np
import pandas as pd
import os
import cv2 
import matplotlib.pyplot as plt
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

from flask import Flask, request, jsonify

app = Flask(__name__)


#By using the <path: url> specifier, we ensure that the string that will come after send-image / is taken as a whole.
@app.route("/send-image/<path:url>")
def image_check(url):
    
    '''
    FUTURE PROCESS
    '''
    
    # When you type http://127.1.0.0:5000/send-image/https://sample-website.com/sample-cdn/photo1.jpg to the browser
    # you will se the whole "https://sample-website.com/sample-cdn/photo1.jpg"
    # return url
    # return jsonify({'amg':'sc'})
    image_path = f'/{url}'
    img = cv2.imread(image_path)
    img = np.expand_dims(img,axis=0)
    # print(img.shape)
    model = load_model('/workspaces/Save_knee/final_model.h5')
    # model.summary()
    y_pred = model.predict(img)
    y_pred = y_pred.tolist()
    idx_c = np.argmax(y_pred)
    name = ['normal','moderate','severe']
    class_name = name[idx_c]

    fi = {'Prediction':y_pred,
          'Class_name':class_name}

    jso = json.dumps(fi)

    return jso



if __name__ == '__main__':
    app.run(debug=True)