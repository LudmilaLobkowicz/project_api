from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import joblib
#from PIL import Image
#import PIL
#import tensorflow as tf
#from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import models
#from tensorflow.keras.backend import expand_dims
#import os


PATH_TO_LOCAL_MODEL = "save_model.pd"
WS_BUCKET_TEST_PATH = "TO be filled in"
BUCKET_NAME = "To be added"

app = FastAPI()

@app.get("/")
def hello():
    return {'message': 'hello word' }

@app.get("/predict_CXray")
def predict(image):
    # compute `prediction` for `X_ray`
    file_path = os.path.join('WS_BUCKET_TEST_PATH')# path to our input
    image = Image.open(file_path)
    image = image.resize((256, 256)) # potentially done by vgg16
    X_test = np.array(image) # transforming image into array

    #X_test=np.stack([X_test]*3, axis=-1) #expansion of dimension -> seems to be already colored
    X_test=tf.keras.applications.vgg16.preprocess_input(X_test, data_format=None)# to preprocess input for vgg16
    X_test= expand_dims(X_test, axis=0) # to add another dimension needed for input (None, 256, 256, 3)

    loaded_model = models.load_model('PATH_TO_LOCAL_MODEL')# loading model with tensorflow

    pred = loaded_model.predict(X_test, verbose=1)[1] # we have to still decide what metrics to use

    return {"prediction": pred}
