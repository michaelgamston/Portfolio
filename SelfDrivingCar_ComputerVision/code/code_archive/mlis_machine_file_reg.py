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


image_size = (244,244)
input_shape = (244,244,3)
batch_size = 64
dropout = 0.3

#define angle CNN
pretrained_model = tf.keras.applications.MobileNet(input_shape=input_shape, include_top=False, weights='imagenet') #We now dont want randomized weights but to load weights from imagenet
pretrained_model.trainable = False # freeze the first layers to the imagenet weights



class Speed_CNN():
    
    def __init__(self):
        
        #define callbacks for both models 
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('/tmp/checkpoint.keras', monitor='val_loss', save_best_only=True)
    
        #import training and val data for the speed CNN
        train_speed, val_speed = tf.keras.preprocessing.image_dataset_from_directory('data/speed/', 
                                                                            labels = 'inferred', 
                                                                            label_mode = 'categorical', #! binary ???
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
        
        #compile train and save model 
        model_speed.compile(optimizer='adam',
                    loss=losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])
        model_speed.fit(train_speed, validation_data=val_speed, epochs=50, callbacks=[callback, checkpoint])
        model_speed.save('model_speed.model.keras')
        

class Angle_CNN(): 
    
    def __init__(self):
        
        #define callbacks for both models 
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('/tmp/checkpoint.keras', monitor='val_loss', save_best_only=True)
        
        #get training and validation data for angle CNN
        train_angle, val_angle = tf.keras.preprocessing.image_dataset_from_directory('data/angle/', 
                                                                            labels = 'inferred', 
                                                                            label_mode = 'int',
                                                                            batch_size = batch_size,
                                                                            image_size = image_size,
                                                                            validation_split = 0.2,
                                                                            subset ='both', 
                                                                            seed = 42)
        
        
        normalization_layer = layers.Rescaling(1./16)
        train_angle = train_angle.map(lambda x, y: (x, tf.cast(normalization_layer(tf.expand_dims(y, -1)), tf.float32)))
        val_angle = val_angle.map(lambda x, y: (x, tf.cast(normalization_layer(tf.expand_dims(y, -1)), tf.float32)))
        

        model_angle = tf.keras.Sequential([
            layers.Rescaling(1./255, input_shape=input_shape),
            pretrained_model,
            layers.GlobalAveragePooling2D(), 
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.Dense(1, activation='sigmoid')
        ])

        model_angle.build()

        #compile train and save model 
        model_angle.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                    loss=losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.MeanSquaredError()])
        model_angle.fit(train_angle, validation_data=val_angle, epochs=50, callbacks=[callback, checkpoint])
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
            
            angle_pred = angle_predictions[index][0]
            #angle_pred = min(categories_angle, key=lambda x: abs(x - angle_pred))
            #angle_pred = categories_angle[np.argmax(angle_predictions[index])]
            speed_pred = categories_speed[np.argmax(speed_predictions[index])]
            
            predictions['speed'].append(speed_pred)
            predictions['angle'].append(angle_pred)
            predictions['image_id'].append(re.findall(r'\d+', paths[index])[0]) #re.split(r'[/.]', paths[index])[2]) 
            
        #save predictions and as a csv 
        df = pd.DataFrame(predictions)
        df.to_csv('outputs.csv', index = False)
        
if __name__ == "__main__":
    Predictor()