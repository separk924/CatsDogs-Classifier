"""
This file creates, trains, and saves a neural network model for distinguishing
photos between cats and dogs.

"""
__author__ = "Seung Park"
__license__ = "University of Puget Sound"
__date__ = "April 24, 2023"


################################################################
########################### IMPORTS ############################
################################################################
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import pandas as pd

############## GET ALL FILES ##############
data_dir = os.listdir('./cats-and-dogs/')
data = {}
id_list = []
label_list = []
for file_name in data_dir:
    label = file_name[:1]
    id = file_name[1:]
    # print(label)
    if label == 'c' or label == 'd':
        if 'id' not in data.keys():
            id_list.append(file_name)
            data['id'] = id_list
        else:
            data.get('id').append(file_name)
        if 'label' not in data.keys():
            label_list.append(label)
            data['label'] = label_list
        else:
            data.get('label').append(label)

############## CONVERT TO PANDAS DATAFRAME ##############
df = pd.DataFrame(data)
df_label_id = df[['id', 'label']]

############## CREATE IMAGE DATA GENERATOR ##############
imageDataGenerator = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=40,
    fill_mode='nearest',
    shear_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.4,1.5])

############## GET ALL IMAGE INFORMATION ##############
train_iterator = imageDataGenerator.flow_from_dataframe(
    dataframe=df_label_id,
    directory='./cats-and-dogs/',
    x_col='id',
    y_col='label',
    target_size=(150, 150),
    batch_size=225,
    class_mode='binary',
    color_mode='rgb')

############## CREATE MODEL ##############
model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPool2D((2, 2)),
    
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

############## SAVE CHECKPOINTS TO PLOT HISTORY ##############
path_checkpoint = 'training_chkpts/cp.ckpt'
directory_checkpoint = os.path.dirname(path_checkpoint)
callback = keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,
                                                 save_weights_only=True,
                                                 verbose=1)

############## TRAIN MODEL ##############
history = model.fit(train_iterator,
                    steps_per_epoch=100,
                    epochs=60,
                    callbacks=[callback])

def plot_result(history):
    acc = history.history['accuracy']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.title('Training accuracy')
    plt.legend()
    plt.figure()

    loss = history.history['loss']
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()
    
plot_result(history)

############## SAVE THE MODEL ##############
model.save("seung.h5", include_optimizer = False)