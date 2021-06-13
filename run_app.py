#!/usr/bin/env python
# coding: utf-8

# In[2]:


from load_model import *
from face_detecter import *
from extract_bottleneck_features import *
from flask import request, url_for, Flask, render_template
from werkzeug.utils import secure_filename
import os
import sys
import tensorflow as tf
from keras.preprocessing import image 
from keras.applications.resnet50 import preprocess_input
import numpy as np
import io
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
        print(area,gender)
    return render_template("skintonepredict.html", gender=gender,area=area)

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/predictskin', methods=['POST'])
def predictskin():
    if request.method == 'POST':
        #gender = request.form["gender"]
        #area = request.form["area"]
        output = "melanoma detected"
    return render_template("predictskin.html", output=output)


"""
global graph, INCEPTION_model
graph = tf.compat.v1.get_default_graph

print("loading Inception Model ......")
INCEPTION_model = load_model()

print("Inception Model Loaded!")
    
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    '''
    args: a path to the image
    returns: a 4D tensor
    '''
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)    
      
    
def INCEPTIONV3_predict_breed(img):
    '''
    args - a preprocessed image with a 4D tensor shape
    returns - dog breed based on the exisiting 133 breeds from ImageNet
             data type: string
    '''
    dog_names = ['Affenpinscher',
 'Afghan hound',
 'Airedale terrier',
 'Akita',
 'Alaskan malamute',
 'American eskimo dog',
 'American foxhound',
 'American staffordshire terrier',
 'American water spaniel',
 'Anatolian shepherd dog',
 'Australian cattle dog',
 'Australian shepherd',
 'Australian terrier',
 'Basenji',
 'Basset hound',
 'Beagle',
 'Bearded collie',
 'Beauceron',
 'Bedlington terrier',
 'Belgian malinois',
 'Belgian sheepdog',
 'Belgian tervuren',
 'Bernese mountain dog',
 'Bichon frise',
 'Black and tan coonhound',
 'Black russian terrier',
 'Bloodhound',
 'Bluetick coonhound',
 'Border collie',
 'Border terrier',
 'Borzoi',
 'Boston terrier',
 'Bouvier des flandres',
 'Boxer',
 'Boykin spaniel',
 'Briard',
 'Brittany',
 'Brussels griffon',
 'Bull terrier',
 'Bulldog',
 'Bullmastiff',
 'Cairn terrier',
 'Canaan dog',
 'Cane corso',
 'Cardigan welsh corgi',
 'Cavalier king charles spaniel',
 'Chesapeake bay retriever',
 'Chihuahua',
 'Chinese crested',
 'Chinese shar-pei',
 'Chow chow',
 'Clumber spaniel',
 'Cocker spaniel',
 'Collie',
 'Curly-coated retriever',
 'Dachshund',
 'Dalmatian',
 'Dandie dinmont terrier',
 'Doberman pinscher',
 'Dogue de bordeaux',
 'English cocker spaniel',
 'English setter',
 'English springer spaniel',
 'English toy spaniel',
 'Entlebucher mountain dog',
 'Field spaniel',
 'Finnish spitz',
 'Flat-coated retriever',
 'French bulldog',
 'German pinscher',
 'German shepherd dog',
 'German shorthaired pointer',
 'German wirehaired pointer',
 'Giant schnauzer',
 'Glen of imaal terrier',
 'Golden retriever',
 'Gordon setter',
 'Great dane',
 'Great pyrenees',
 'Greater swiss mountain dog',
 'Greyhound',
 'Havanese',
 'Ibizan hound',
 'Icelandic sheepdog',
 'Irish red and white setter',
 'Irish setter',
 'Irish terrier',
 'Irish water spaniel',
 'Irish wolfhound',
 'Italian greyhound',
 'Japanese chin',
 'Keeshond',
 'Kerry blue terrier',
 'Komondor',
 'Kuvasz',
 'Labrador retriever',
 'Lakeland terrier',
 'Leonberger',
 'Lhasa apso',
 'Lowchen',
 'Maltese',
 'Manchester terrier',
 'Mastiff',
 'Miniature schnauzer',
 'Neapolitan mastiff',
 'Newfoundland',
 'Norfolk terrier',
 'Norwegian buhund',
 'Norwegian elkhound',
 'Norwegian lundehund',
 'Norwich terrier',
 'Nova scotia duck tolling retriever',
 'Old english sheepdog',
 'Otterhound',
 'Papillon',
 'Parson russell terrier',
 'Pekingese',
 'Pembroke welsh corgi',
 'Petit basset griffon vendeen',
 'Pharaoh hound',
 'Plott',
 'Pointer',
 'Pomeranian',
 'Poodle',
 'Portuguese water dog',
 'Saint bernard',
 'Silky terrier',
 'Smooth fox terrier',
 'Tibetan mastiff',
 'Welsh springer spaniel',
 'Wirehaired pointing griffon',
 'Xoloitzcuintli',
 'Yorkshire terrier']
    bottleneck_features = extract_InceptionV3(img)
    pred_vector = INCEPTION_model.predict(bottleneck_features)
    return dog_names[np.argmax(pred_vector)]
    

# graph = tf.get_default_graph()


@app.route('/', methods = ['GET'])
def index():
    return render_template("index.html")



@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    
    #a wrapper funtion that uses INCEPTIONV3_predict_breed function
    # to make predictions from the user supplied images

   
    # Get the image file from the post request
    file = request.files['file']
        
        # Save the file with proper image path
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, secure_filename(file.filename))
    file.save(file_path)
    image = path_to_tensor(file_path)
    image = preprocess_input(image)
    
    #if dog_detecting(image):
    with graph.as_default():
        prediction = INCEPTIONV3_predict_breed(image)
        result = str(prediction)
        
    
        # if face_detecting(image):
        #     result = 'This photo looks like a/an {}'.format(prediction)
    
        
        #result = 'So sorry: This image is not recognizable!'
        
        result = "This is very likely a " + result

    # else: 
    #     result = "This doesn't seem to be a dog --- Sorry!"
        
        return result

    
    


    
    



"""

    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port = port)

