import tensorflow as tf 
from keras import layers, models, optimizers, losses, metrics
from keras.regularizers import l2
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

pretrained_model = tf.keras.applications.MobileNet(input_shape=input_shape, include_top=False, weights='imagenet')
pretrained_model.trainable = False # freeze the first layers to the imagenet weights


class Speed_CNN():
    
    def __init__(self):
        # Training and Validation Data
        train_speed, val_speed = tf.keras.preprocessing.image_dataset_from_directory('data/speed/', 
                                                                            labels = 'inferred', 
                                                                            label_mode = 'categorical',
                                                                            batch_size = batch_size,
                                                                            image_size = image_size,
                                                                            validation_split = 0.2,
                                                                            subset ='both', 
                                                                            seed = 42)
        
        # Augmentation
        augmentation_layers = tf.keras.Sequential([
            #tf.keras.layers.RandomTranslation(height_factor=(0, 0.05), width_factor=(0.025, 0.025)), #translation is risky.
            tf.keras.layers.RandomBrightness(factor=(-0.3, 0.3), value_range=(0, 255)),
            tf.keras.layers.RandomSharpness(factor=(0.3,0.7), value_range=(0,255)),
            #layers.RandomSaturation(factor=(0.35,0.65), value_range=(0,255)),  #unavailable in current keras. 
            tf.keras.layers.RandomContrast(factor=(0.0,0.2), value_range=(0,255))
        ])
        augmented_train_speed = train_speed.map(lambda image, label: (augmentation_layers(image, training=True), label))
        train_speed = train_speed.concatenate(augmented_train_speed)  # APPEND augmented data
        
        # CNN
        model_speed = tf.keras.Sequential([
            layers.Rescaling(1./255, input_shape=input_shape),
            pretrained_model,
            layers.GlobalAveragePooling2D(), 
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(2, activation='softmax')
        ])
        
        # Callback and Checkpoint
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('/tmp/checkpoint.keras', monitor='val_loss', save_best_only=True)
        
        # Build, Compile, Train, and Save model
        model_speed.build()
        model_speed.compile(optimizer='adam',
                    loss=losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])
        model_speed.fit(train_speed, validation_data=val_speed, epochs=50, callbacks=[callback, checkpoint])
        model_speed.save('model_speed.model.keras')
        


class Angle_CNN(): 
    
    def __init__(self):
        
        # Training and Validation Data
        train_angle, val_angle = tf.keras.preprocessing.image_dataset_from_directory('data/angle/', 
                                                                            labels = 'inferred', 
                                                                            label_mode = 'categorical',
                                                                            batch_size = batch_size,
                                                                            image_size = image_size,
                                                                            validation_split = 0.2,
                                                                            subset ='both', 
                                                                            seed = 42)

        # Augmentation
        augmentation_layers = tf.keras.Sequential([
            #tf.keras.layers.RandomTranslation(height_factor=(0, 0.05), width_factor=(0.025, 0.025)), #translation is risky.
            tf.keras.layers.RandomBrightness(factor=(-0.3, 0.3), value_range=(0, 255)),
            tf.keras.layers.RandomSharpness(factor=(0.3,0.7), value_range=(0,255)),
            #layers.RandomSaturation(factor=(0.35,0.65), value_range=(0,255)),  #unavailable in current keras. 
            tf.keras.layers.RandomContrast(factor=(0.0,0.2), value_range=(0,255))
        ])
        augmented_train_angle = train_angle.map(lambda image, label: (augmentation_layers(image, training=True),label))
        train_angle = train_angle.concatenate(augmented_train_angle)  # APPEND augmented data
        
        class_weights = self.compute_class_weights(train_angle)
        
        # CNN
        model_angle = tf.keras.Sequential([
            layers.Rescaling(1./255, input_shape=input_shape),
            pretrained_model,
            layers.GlobalAveragePooling2D(), 
            layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.Dense(17, activation='softmax')
        ])

        # Callback & Checkpoint
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('/tmp/checkpoint.keras', monitor='val_loss', save_best_only=True)
        
        # Build, Compile, Train, and Save model
        model_angle.build()
        model_angle.compile(optimizer='adam',
                    loss=losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])
        model_angle.fit(train_angle, validation_data=val_angle, epochs=40, callbacks=[callback, checkpoint], class_weight=class_weights)
        
        #! Retrain pretrained model : Why is data regathered - should the same data be used from above instead (espescially due to augmentation use)
        pretrained_model.trainable = True

        # It's important to recompile your model after you make any changes
        # to the `trainable` attribute of any inner layer, so that your changes
        # are take into account
        model_angle.compile(optimizer=optimizers.Adam(1e-5),# Very low learning rate
                    loss=losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])
        
        train_angle_ret, val_angle_ret = tf.keras.preprocessing.image_dataset_from_directory('data/angle/', 
                                                                            labels = 'inferred', 
                                                                            label_mode = 'categorical',
                                                                            batch_size = batch_size,
                                                                            image_size = image_size,
                                                                            validation_split = 0.2,
                                                                            subset ='both', 
                                                                            seed = 666)
        

        augmented_train_angle_ret = train_angle_ret.map(lambda image, label: (augmentation_layers(image, training=True),label))
        train_angle_ret = train_angle_ret.concatenate(augmented_train_angle_ret)  # APPEND augmented data
        
        #redefine callbacks for both models 
        callback_rebuild = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint_rebuild = tf.keras.callbacks.ModelCheckpoint('/tmp/checkpoint.keras', monitor='val_loss', save_best_only=True)

        # Train end-to-end. Be careful to stop before you overfit!
        model_angle.fit(train_angle_ret, validation_data=val_angle_ret, epochs=20, callbacks=[callback_rebuild, checkpoint_rebuild], class_weight=class_weights)                            
        model_angle.save('model_angle.model.keras')

    def compute_class_weights(self, dataset):
        """
        Compute class weights to handle class imbalance
        
        Args:
            dataset: TensorFlow dataset with categorical labels
        
        Returns:
            Dictionary of class weights
        """
        # Collect all labels
        labels = []
        for _, label in dataset:
            labels.append(tf.argmax(label, axis=1).numpy())
        
        # Flatten labels
        labels = np.concatenate(labels)
        
        # Count occurrences of each class
        unique, counts = np.unique(labels, return_counts=True)
        
        # Compute weights
        total_samples = len(labels)
        class_weights = {i: total_samples / (len(unique) * count) for i, count in zip(unique, counts)}
        
        # Normalize weights
        max_weight = max(class_weights.values())
        class_weights = {k: v/max_weight for k, v in class_weights.items()}
        
        print("Class Weights:", class_weights)
        return class_weights


class Predictor():
    
    def __init__(self):
        #self.run_speed_cnn()
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
            predictions['image_id'].append(re.findall(r'\d+', paths[index])[0]) #ALT: re.split(r'[/.]', paths[index])[2]) 
            
        #save predictions and as a csv 
        df = pd.DataFrame(predictions)
        df.to_csv('outputs.csv', index = False)

      
if __name__ == "__main__":
    Predictor()