import tensorflow as tf 
from keras import layers, models, optimizers, losses, metrics
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd 
import os
import argparse
from sklearn.metrics import accuracy_score
import re
import json

#! ****** Importing the training logger class from the Logger module ****** !#

from Logger import TrainingLogger     # Ensure this file is in the same directory as the Logger module / file

#! ************************************************************************ !#

image_size = (244,244)
input_shape = (244,244,3)
batch_size = 64
dropout = 0.2

#define angle CNN
pretrained_model = tf.keras.applications.VGG19(input_shape=input_shape, include_top=False, weights='imagenet') #We now dont want randomized weights but to load weights from imagenet
pretrained_model.trainable = False # freeze the first layers to the imagenet weights



class Speed_CNN():
    
    def __init__(self):
    
        #import training and val data for the speed CNN
        train_speed, val_speed = tf.keras.preprocessing.image_dataset_from_directory('data/speed/', 
                                                                            labels = 'inferred', 
                                                                            label_mode = 'categorical',
                                                                            batch_size = batch_size,
                                                                            image_size = image_size,
                                                                            validation_split = 0.2,
                                                                            subset ='both', 
                                                                            seed = 42)
        
        model_speed = tf.keras.Sequential([
            layers.Rescaling(1./255, input_shape=input_shape),
            pretrained_model,
            layers.GlobalAveragePooling2D(), 
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(2, activation='softmax')
        ])
        
        
        #define callbacks for both models 
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('/tmp/checkpoint.keras', monitor='val_loss', save_best_only=True)
        
        #! ****** Create log_callback for speed model ****** !#
        
        log_callback = TrainingLogger(dict_name='speed')

        #! ************************************************* !#
        
        #compile train and save model 
        model_speed.compile(optimizer='adam',
                    loss=losses.CategoricalCrossentropy(),
                    metrics=['Accuracy', 'MeanSquaredError', 'MeanAbsoluteError', 'Precision', 'Recall'])
        model_speed.fit(train_speed, validation_data=val_speed, epochs=50, callbacks=[callback, checkpoint, log_callback])
        model_speed.save('model_speed.model.keras')
        

class Angle_CNN(): 
    
    def __init__(self):
        
        #get training and validation data for angle CNN
        train_angle, val_angle = tf.keras.preprocessing.image_dataset_from_directory('data/angle/', 
                                                                            labels = 'inferred', 
                                                                            label_mode = 'categorical',
                                                                            batch_size = batch_size,
                                                                            image_size = image_size,
                                                                            validation_split = 0.2,
                                                                            subset ='both', 
                                                                            seed = 42)


        model_angle = tf.keras.Sequential([
            layers.Rescaling(1./255, input_shape=input_shape),
            pretrained_model,
            layers.GlobalAveragePooling2D(), 
            layers.Dense(640, activation='relu'),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.Dense(17, activation='softmax')
        ])


        #define callbacks for both models 
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('/tmp/checkpoint.keras', monitor='val_loss', save_best_only=True)
        
        #! ****** Create log_callback for angle model ****** !#
        
        log_callback = TrainingLogger(dict_name='angle')
        
        #! ************************************************* !#
        
        #compile train and save model 
        model_angle.build()
        model_angle.compile(optimizer='adam',
                    loss=losses.CategoricalCrossentropy(),
                    metrics=['Accuracy', 'MeanSquaredError', 'MeanAbsoluteError', 'Precision', 'Recall'])
        model_angle.fit(train_angle, validation_data=val_angle, epochs=150, callbacks=[callback, checkpoint, log_callback])
        
        # Unfreeze the base model
        pretrained_model.trainable = True

        # It's important to recompile your model after you make any changes
        # to the `trainable` attribute of any inner layer, so that your changes
        # are take into account
        model_angle.compile(optimizer=optimizers.Adam(1e-5),# Very low learning rate
                    loss=losses.CategoricalCrossentropy(),
                    metrics=['Accuracy', 'MeanSquaredError', 'MeanAbsoluteError', 'Precision', 'Recall'])
        
        train_angle, val_angle = tf.keras.preprocessing.image_dataset_from_directory('data/angle/', 
                                                                            labels = 'inferred', 
                                                                            label_mode = 'categorical',
                                                                            batch_size = batch_size,
                                                                            image_size = image_size,
                                                                            validation_split = 0.2,
                                                                            subset ='both', 
                                                                            seed = 666)
        #redefine callbacks for both models 
        callback_rebuild = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        checkpoint_rebuild = tf.keras.callbacks.ModelCheckpoint('/tmp/checkpoint.keras', monitor='val_loss', save_best_only=True)
        
        #! ****** Create log_callback for angle_rebuild model ****** !#
        
        log_callback_rebuild = TrainingLogger(dict_name='angle_rebuild')

        #! ********************************************************* !#

        # Train end-to-end. Be careful to stop before you overfit!
        model_angle.fit(train_angle, epochs=20, callbacks=[callback_rebuild, checkpoint_rebuild, log_callback_rebuild], validation_data=val_angle)                            

        model_angle.save('model_angle.model.keras')

class Predictor():
    
    def __init__(self):
        self.run_speed_cnn()
        self.run_angle_cnn()
        self.get_predictions()
    
    def run_speed_cnn(self):
        cnn = Speed_CNN()
        
    def run_angle_cnn(self):
        cnn = Angle_CNN()
        
    def get_predictions(self):
        # load speed model for prediction
        model_speed = models.load_model('model_speed.model.keras')
        model_angle = models.load_model('model_angle.model.keras')

        #create dictionary for tracking predictions
        predictions = {'image_id' : [], 'angle' : [], 'speed' : []}

        #categories for prediction, speed and angle 
        categories_angle = [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0]
        categories_speed = [0, 1]


        test_images = tf.keras.preprocessing.image_dataset_from_directory('data/test_data/', 
                                                                            labels = None, 
                                                                            label_mode = None,
                                                                            batch_size = 64,
                                                                            image_size = image_size,
                                                                            shuffle = False)
        
        paths = test_images.file_paths

        angle_predictions = model_angle.predict(test_images)
        speed_predictions = model_speed.predict(test_images)

        #loop through test data and produce predictions 
        for index in range(len(angle_predictions)):
            
            
            angle_pred = categories_angle[np.argmax(angle_predictions[index])]
            speed_pred = categories_speed[np.argmax(speed_predictions[index])]
            
            predictions['speed'].append(speed_pred)
            predictions['angle'].append(angle_pred)
            predictions['image_id'].append(re.split(r'[/.]', paths[index])[2])
            
        #save predictions and as a csv 
        df = pd.DataFrame(predictions)
        df.to_csv('outputs.csv', index = False)
        
if __name__ == "__main__":
    Predictor()