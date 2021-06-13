from flask import request, url_for, Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import sys
import tensorflow as tf
import numpy as np
import io
from src.skin_color import estimate_skin
from src.infer import predict, process_preds
from PIL import Image

app = Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html")
    #return render_template('/Users/Neeral/hackathonproj/hackapp/keras_image_classifer/templates/NewBiz/index.html')

@app.route('/skintone')
def skintone():
    return render_template("skintone.html")

@app.route('/skintonepredict', methods=['POST'])
def skintonepredict():
    if request.method == 'POST':
        gender = request.form["gender"]
        area = request.form["area"]
        age = request.form["age"]
        print(area,gender)
    return render_template("skintonepredict.html", gender=gender,area=area, age=age)

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/predictskin', methods=['POST'])
def predictskin():
    imagefile=request.files["imagefile"]
    image_path= "./templates/" + imagefile.filename
    imagefile.save(image_path)
    result = predict(image_path)

    return render_template('predict.html', result=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port = port)

