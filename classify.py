"""
This file uploads a model and makes predictions for whether the photo contains cats
or dogs

"""
__author__ = "Seung Park"
__license__ = "University of Puget Sound"
__date__ = "April 24, 2023"


################################################################
########################### IMPORTS ############################
################################################################
import sys
import tensorflow as tf
from tensorflow import keras 
from keras.utils import load_img
from keras.utils import img_to_array

############## GET COMMAND LINE ARGS ##############
neural_net = sys.argv[1]
image_names = sys.argv[1:]

############## CONVERT IMAGES FOR PREDICTIONS ##############
def convert_images(images):
    new_images = []
    for item in images:
        image = load_img(item, target_size=(150,150))
        image = img_to_array(image)
        image = image.reshape(1,150,150,3)
        new_images.append(image)
    return new_images
images = convert_images(image_names)

############## LOAD MODEL ##############
model = keras.models.load_model('./seung.hd5')

############## MAKE PREDICTIONS ##############
i = 0
for item in images:
    prediction = model.predict(item)
    if prediction == 1:
        print('IMAGE NAME:', image_names[i], 'PREDICTION: Dog')
    else:
        print('IMAGE NAME:', image_names[i], 'PREDICTION: Cat')
    i+=1