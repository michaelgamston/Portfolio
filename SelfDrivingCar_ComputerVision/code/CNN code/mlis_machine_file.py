import tensorflow as tf 
from keras import layers, models, optimizers, losses
import tensorflow as tf
import numpy as np
import random
import pandas as pd 
import re
import json


image_size = (244,244)
input_shape = (244,244,3)
batch_size = 64
dropout = 0.3

#define angle CNN
pretrained_model = tf.keras.applications.MobileNet(input_shape=input_shape, include_top=False, weights='imagenet') #We now dont want randomized weights but to load weights from imagenet
pretrained_model.trainable = False # freeze the first layers to the imagenet weights




class TrainingLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.log_file = 'log_file.json'
        self.history = {"epoch": [], "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history["epoch"].append(epoch + 1)
        self.history["loss"].append(logs.get('loss'))
        self.history["accuracy"].append(logs.get('accuracy'))
        self.history["val_loss"].append(logs.get('val_loss'))
        self.history["val_accuracy"].append(logs.get('val_accuracy'))



class Speed_CNN():
    
    def __init__(self):
        
        #define callbacks for both models 
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('/tmp/checkpoint.keras', monitor='val_loss', save_best_only=True)
        log_callback1 = TrainingLogger('speed_training_log1.json')
        log_callback2 = TrainingLogger('speed_training_log2.json')
    
        #import training and val data for the speed CNN
        train_speed, val_speed = tf.keras.preprocessing.image_dataset_from_directory('data/speed_big/', 
                                                                            labels = 'inferred', 
                                                                            label_mode = 'categorical',
                                                                            batch_size = batch_size,
                                                                            image_size = image_size,
                                                                            validation_split = 0.2,
                                                                            subset ='both', 
                                                                            seed = 42)
        
        #train_speed_hsv = train_speed.map(lambda image, label: (ops.image.rgb_to_hsv(image), label))
        #val_speed_hsv = val_speed.map(lambda image, label: (ops.image.rgb_to_hsv(image), label))
        
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
        model_speed.fit(train_speed, validation_data=val_speed, epochs=50, callbacks=[callback, checkpoint, log_callback1])
        model_speed.save('model_speed.model.keras')
        
        # Unfreeze the base model
        pretrained_model.trainable = True

        # It's important to recompile your model after you make any changes
        # to the `trainable` attribute of any inner layer, so that your changes
        # are take into account
        model_speed.compile(optimizer=optimizers.Adam(1e-5),# Very low learning rate
                    loss=losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])
        
        #define callbacks for both models 
        callback_rebuild = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        checkpoint_rebuild = tf.keras.callbacks.ModelCheckpoint('/tmp/checkpoint.keras', monitor='val_loss', save_best_only=True)
        
        train_speed, val_speed = tf.keras.preprocessing.image_dataset_from_directory('data/speed/', 
                                                                            labels = 'inferred', 
                                                                            label_mode = 'categorical',
                                                                            batch_size = batch_size,
                                                                            image_size = image_size,
                                                                            validation_split = 0.2,
                                                                            subset ='both', 
                                                                            seed = 666)
        
        #train_speed_hsv = train_speed.map(lambda image, label: (ops.image.rgb_to_hsv(image), label))
        #val_speed_hsv = val_speed.map(lambda image, label: (ops.image.rgb_to_hsv(image), label))
        
    
        #redefine callbacks for both models 
        callback_rebuild = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        checkpoint_rebuild = tf.keras.callbacks.ModelCheckpoint('/tmp/checkpoint.keras', monitor='val_loss', save_best_only=True)

        # Train end-to-end. Be careful to stop before you overfit!
        model_speed.fit(train_speed, epochs=20, callbacks=[callback_rebuild, checkpoint_rebuild, log_callback2], validation_data=val_speed)    




class Angle_CNN(): 
    
    def __init__(self):
        #get training and validation data for angle CNN
        train_angle, val_angle = tf.keras.preprocessing.image_dataset_from_directory('data/angle_big/', 
                                                                            labels = 'inferred', 
                                                                            label_mode = 'categorical',
                                                                            batch_size = batch_size,
                                                                            image_size = image_size,
                                                                            validation_split = 0.2,
                                                                            subset ='both', 
                                                                            seed = 42)
        
        class_weights = self.compute_class_weights(train_angle)
        
        # train_angle_hsv = train_angle.map(lambda image, label : (ops.image.rgb_to_hsv(image), label))
        # val_angle_hsv = val_angle.map(lambda image, label : (ops.image.rgb_to_hsv(image), label)) 

        #define callbacks for both models 
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('/tmp/checkpoint.keras', monitor='val_loss', save_best_only=True)
        log_callback1 = TrainingLogger('angle_training_log1.json')
        log_callback2 = TrainingLogger('angle_training_log2.json')

        model_angle = tf.keras.Sequential([
            layers.Rescaling(1./255, input_shape=input_shape),
            pretrained_model,
            layers.GlobalAveragePooling2D(), 
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.Dense(17, activation='softmax')
        ])

        model_angle.build()

        #compile train and save model 
        model_angle.compile(optimizer='adam',
                    loss=losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])
        model_angle.fit(train_angle, validation_data=val_angle, epochs=150, callbacks=[callback, checkpoint, log_callback1], class_weight=class_weights)
        
        # Unfreeze the base model
        pretrained_model.trainable = True

        # It's important to recompile your model after you make any changes
        # to the `trainable` attribute of any inner layer, so that your changes
        # are take into account
        model_angle.compile(optimizer=optimizers.Adam(1e-5),# Very low learning rate
                    loss=losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])
        
        train_angle, val_angle = tf.keras.preprocessing.image_dataset_from_directory('data/angle/', 
                                                                            labels = 'inferred', 
                                                                            label_mode = 'categorical',
                                                                            batch_size = batch_size,
                                                                            image_size = image_size,
                                                                            validation_split = 0.2,
                                                                            subset ='both', 
                                                                            seed = 666)
        
        #train_angle_hsv = train_angle.map(lambda image, label : (ops.image.rgb_to_hsv(image), label))
        #val_angle_hsv = val_angle.map(lambda image, label : (ops.image.rgb_to_hsv(image), label)) 
        
        #redefine callbacks for both models 
        callback_rebuild = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        checkpoint_rebuild = tf.keras.callbacks.ModelCheckpoint('checkpoint.keras', monitor='val_loss', save_best_only=True)

        # Train end-to-end. Be careful to stop before you overfit!
        model_angle.fit(train_angle, epochs=10, callbacks=[callback_rebuild, checkpoint_rebuild, log_callback2], validation_data=val_angle, class_weight=class_weights)                            

        model_angle.save('model_angle.model.keras')
        
        
    def compute_class_weights(self, dataset):  #! AI generated 
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
        #self.run_angle_cnn()
        #self.run_regression_CNN()
        #self.run_multi_head_CNN()
        self.get_predictions()
    
    def run_speed_cnn(self):
        cnn = Speed_CNN()
        
    def run_angle_cnn(self):
        cnn = Angle_CNN()
        
    def run_regression_CNN(self):
        cnn = Regression_CNN()
        
    def run_multi_head_CNN(self):
        cnn = Multi_head_CNN()
        
    def get_predictions(self):
        # load speed model for prediction
        # model_speed = models.load_model('model_speed.model.keras')
        # model_angle = models.load_model('model_angle.model.keras')
        model_regression = models.load_model('multi_head_model_reg.h5')
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

        # angle_predictions = model_angle.predict(test_images)
        # speed_predictions = model_speed.predict(test_images)
        
        angle_pred, speed_pred = model_regression.predict(test_images)
        
        

        #loop through test data and produce predictions 
        for index, preds in enumerate(angle_pred):
            
            print(preds)
            print(speed_pred[0])
            input()
            # angle_pred = categories_angle[np.argmax(angle_predictions[index])]
            # speed_pred = categories_speed[np.argmax(speed_predictions[index])]
            
            predictions['speed'].append(speed_pred)
            predictions['angle'].append(angle_pred)
            predictions['image_id'].append(re.findall(r'\d+', paths[index])[-1])
            
        #save predictions and as a csv 
        df = pd.DataFrame(predictions)
        df.to_csv('outputs.csv', index = False)
        
if __name__ == "__main__":
    Predictor()
    
    




