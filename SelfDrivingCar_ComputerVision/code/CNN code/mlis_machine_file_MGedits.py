import tensorflow as tf 
from keras import layers, models, optimizers, losses
import tensorflow as tf
import numpy as np
import random
import pandas as pd 
import re   
import os
from keras.layers import Input
from keras.utils import to_categorical
import cv2 as cv
from Logger import TrainingLogger
import json

image_size = (244,244)
input_shape = (244,244,3)
batch_size = 64
dropout = 0.2

# #define angle CNN
pretrained_model = tf.keras.applications.MobileNet(input_shape=input_shape, include_top=False, weights='imagenet') #We now dont want randomized weights but to load weights from imagenet
pretrained_model.trainable = False # freeze the first layers to the imagenet weights



class Speed_CNN():
    
    def __init__(self):
        
        logger = TrainingLogger(dict_name='speed', log_file='speed.json')
        #define callbacks for both models 
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('checkpoint.keras', monitor='val_loss', save_best_only=True)
    
        #import training and val data for the speed CNN
        train_speed, val_speed = tf.keras.preprocessing.image_dataset_from_directory('data/big_data/speed/', 
                                                                            labels = 'inferred', 
                                                                            label_mode = 'binary',
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
                    loss=losses.BinaryCrossentropy(),
                    metrics=['Accuracy', 'MeanSquaredError', 'Recall', 'Precision'])
        model_speed.fit(train_speed, validation_data=val_speed, epochs=50, callbacks=[callback, checkpoint, logger])
        model_speed.save('model_speed.h5')
        

class Angle_CNN(): 
    
    def __init__(self):
        #get training and validation data for angle CNN
        train_angle, val_angle = tf.keras.preprocessing.image_dataset_from_directory('data/big_data/angle/', 
                                                                            labels = 'inferred', 
                                                                            label_mode = 'categorical',
                                                                            batch_size = batch_size,
                                                                            image_size = image_size,
                                                                            validation_split = 0.2,
                                                                            subset ='both', 
                                                                            seed = 42)

        #define callbacks for both models 
        logger = TrainingLogger(dict_name='angle', log_file='angle.json')
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('checkpoint.keras', monitor='val_loss', save_best_only=True)

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

        model_angle.build()

        #compile train and save model 
        model_angle.compile(optimizer='adam',
                    loss=losses.CategoricalCrossentropy(),
                    metrics=['Accuracy', 'MeanSquaredError', 'Recall', 'Precision'])
        model_angle.fit(train_angle, validation_data=val_angle, epochs=150, callbacks=[callback, checkpoint, logger])
        
        # # Unfreeze the base model
        # pretrained_model.trainable = True
        
        # # It's important to recompile your model after you make any changes
        # # to the `trainable` attribute of any inner layer, so that your changes
        # # are take into account
        # model_angle.compile(optimizer=optimizers.Adam(1e-5),# Very low learning rate
        #             loss=losses.CategoricalCrossentropy(),
        #             metrics=['Accuracy', 'MeanSquaredError', 'Recall', 'Precision'])
        
        # train_angle, val_angle = tf.keras.preprocessing.image_dataset_from_directory('data/angle/', 
        #                                                                     labels = 'inferred', 
        #                                                                     label_mode = 'categorical',
        #                                                                     batch_size = batch_size,
        #                                                                     image_size = image_size,
        #                                                                     validation_split = 0.2,
        #                                                                     subset ='both', 
        #                                                                     seed = 666)
        
        # #train_angle_hsv = train_angle.map(lambda image, label : (ops.image.rgb_to_hsv(image), label))
        # #val_angle_hsv = val_angle.map(lambda image, label : (ops.image.rgb_to_hsv(image), label)) 
        
        # #redefine callbacks for both models 
        # callback_rebuild = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        # checkpoint_rebuild = tf.keras.callbacks.ModelCheckpoint('checkpoint.keras', monitor='val_loss', save_best_only=True)
        # logger = TrainingLogger(dict_name='angle_retrain')
        

        # # Train end-to-end. Be careful to stop before you overfit!
        # model_angle.fit(train_angle, epochs=10, callbacks=[callback_rebuild, checkpoint_rebuild, logger], validation_data=val_angle)                            
                        

        model_angle.save('model_angle.h5')
  

   
class Regression_CNN(): 
    
    train_file_num = 0
    val_file_num = 0  
    
    def __init__(self):
        
        ignore = [4895, 3999, 8285, 10171, 3141, 3884]
        data_pth = 'data/training_data/training_data/'
        self.image_df = pd.read_csv('data/training_norm.csv')
        
        for id in ignore: 
            self.image_df.drop(self.image_df[self.image_df['image_id'] == id].index, inplace=True)
            
        
        shuffled_df = self.image_df.sample(frac=1).reset_index(drop=True)
        #create a 20% split for training and validation
        split = int(len(shuffled_df)/5)
        train_df = shuffled_df.iloc[split:]
        val_df = shuffled_df.iloc[:split]
        
        
        # Load all image file paths and labels
        trian_image_paths = [f'{data_pth}{i}.png' for i in train_df['image_id'].values]
        train_labels = train_df[['angle', 'speed']].values
        # Create a Dataset object
        train_dataset = tf.data.Dataset.from_tensor_slices((trian_image_paths, train_labels))
        # Map the image loading function to the dataset
        train_dataset = train_dataset.map(self.preprocess)
        # Shuffle and batch the dataset
        train_dataset = train_dataset.batch(batch_size)
        
        # Load all image file paths and labels
        val_image_paths = [f'{data_pth}{i}.png' for i in val_df['image_id'].values]
        val_labels = val_df[['angle', 'speed']].values
        # Create a Dataset object
        val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_labels))
        # Map the image loading function to the dataset
        val_dataset = val_dataset.map(self.preprocess)
        # Shuffle and batch the dataset
        val_dataset = val_dataset.batch(batch_size)
        
        #define callbacks for both models 
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('checkpoint.keras', monitor='val_loss', save_best_only=True)
        logger = TrainingLogger(dict_name='regression')

        
        model_regession = tf.keras.Sequential([
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
            layers.Dense(2, activation='linear')
        ])
        
        # model_regession = tf.keras.Sequential([
        #     layers.Rescaling(1./255, input_shape=input_shape),
        #     layers.Conv2D(64, (3,3)),
        #     layers.MaxPooling2D(),
        #     layers.Conv2D(128, (3,3)),
        #     layers.MaxPooling2D(),
        #     layers.Conv2D(128, (3,3)),
        #     layers.Flatten(), 
        #     layers.Dense(64, activation='relu'),
        #     layers.Dense(2, activation='linear'),
        # ])

        model_regession.build()

        #compile train and save model 
        model_regession.compile(optimizer='adam',
                    loss=losses.MeanSquaredError(),
                    metrics=['Accuracy', 'MeanSquaredError'])
        model_regession.fit(train_dataset, validation_data=val_dataset, epochs=150, callbacks=[callback, checkpoint, logger], batch_size=batch_size)                         

        model_regession.save('model_regression.h5')
    
    def load_image(self, image_path):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.resize(image, [244, 244])  # Resize images
            #image = image / 255.0  # Normalize images to [0, 1]
            return image
        
    def preprocess(self, im_pth, label):
        image = self.load_image(im_pth)
        return image, label
    

TRAIN_TEST_SPLIT = 0.8

class Data_generator():
    
    def __init__(self, df):
        self.df = df 
        
    def generate_split_indexes(self):
        
        p = np.random.permutation(len(self.df))
        
        train_up_to = int(len(self.df) * TRAIN_TEST_SPLIT)
        
        train_idx = p[:train_up_to]
        valid_idx = p[train_up_to:]
        
        return train_idx, valid_idx
    
    def load_image(self, image_path):
            image_colour = cv.imread(image_path)
            image = cv.cvtColor(image_colour, cv.COLOR_BGR2GRAY)
            image = cv.medianBlur(image,5)
            ret,threshold = cv.threshold(image,127,255,cv.THRESH_BINARY)
            contours, heirarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(image_colour, contours, -1, (0,0,255), 2)
            image_colour = cv.resize(image_colour, (244,244))
            
            return image_colour
        
    def generate_images(self, image_idx, is_training, batch_size=16):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        """
        
        # arrays to store our batched data
        images, angle, speed = [], [], []
        while True:
            for idx in image_idx:
                feature = self.df.iloc[idx]
                
                categories_angle = [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0]
                
                speed_val = feature['speed']
                angle_val = categories_angle.index(feature['angle'])
                
                data_pth = 'data/big_data/training_data/training_data/'
                #data_pth = 'data/training_data/training_data/'
                
                im = self.load_image(f"{data_pth}{int(feature['image_id'])}.png")
                
                speed.append(speed_val)
                angle.append(to_categorical(angle_val, len(categories_angle)))
                images.append(im)
                
                # yielding condition
                if len(images) >= batch_size:
                    yield (np.array(images), (np.array(angle), np.array(speed)))
                    images, angle, speed = [], [], []
                    
            if not is_training:
                break
        
class Multi_head_CNN():
    
    def __init__(self):
        
        #get training and validation data for angle CNN
        ignore = [4895, 3999, 8285, 10171, 3141, 3884]
        self.image_df = pd.read_csv('data/big_data/updated_training_norm-15452.csv')
        #self.image_df = pd.read_csv('data/training_norm.csv')
        
        for id in ignore: 
            self.image_df.drop(self.image_df[self.image_df['image_id'] == id].index, inplace=True)
              
        data_generator = Data_generator(self.image_df)
        train_idx, valid_idx = data_generator.generate_split_indexes() 
        
        train_gen = data_generator.generate_images(train_idx, is_training=True, batch_size=batch_size)
        val_gen = data_generator.generate_images(valid_idx, is_training=True, batch_size=batch_size)
        
        # shuffled_df = self.image_df.sample(frac=1).reset_index(drop=True)
        # #create a 20% split for training and validation
        # split = int(len(shuffled_df)/5)
        # train_df = shuffled_df.iloc[split:]
        # val_df = shuffled_df.iloc[:split]
        
        # data_pth = 'data/training_data/training_data/'
        # # Load all image file paths and labels
        # trian_image_paths = [f'{data_pth}{i}.png' for i in train_df['image_id'].values]
        # train_labels = train_df[['angle', 'speed']].values
        # # Create a Dataset object
        # train_dataset = tf.data.Dataset.from_tensor_slices((trian_image_paths, train_labels))
        # # Map the image loading function to the dataset
        # train_dataset = train_dataset.map(self.preprocess)
        # # Shuffle and batch the dataset
        # train_dataset = train_dataset.batch(batch_size)
        
        # # Load all image file paths and labels
        # val_image_paths = [f'{data_pth}{i}.png' for i in val_df['image_id'].values]
        # val_labels = val_df[['angle', 'speed']].values
        # # Create a Dataset object
        # val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_labels))
        # # Map the image loading function to the dataset
        # val_dataset = val_dataset.map(self.preprocess)
        # # Shuffle and batch the dataset
        # val_dataset = val_dataset.batch(batch_size)
            
        #define callbacks for both models 
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('checkpoint.keras', monitor='val_loss', save_best_only=True)
        logger = TrainingLogger(dict_name='multi_head_post')
        
        inputs = Input(shape=input_shape)
        angle_head = self.get_angle_head(inputs)
        speed_head = self.get_speed_head(inputs)
        
        model = models.Model(inputs=inputs,
                     outputs = [angle_head, speed_head],
                     name="better_than_a_tesla")
        
        model.build(input_shape)

        #compile train and save model 
        model.compile(optimizer='adam',
                    loss={
                        'speed_output': 'binary_crossentropy',
                        'angle_output': 'mse', 
                        },
                    metrics={
                        'speed_output': ['Accuracy', 'MeanSquaredError', 'Recall', 'Precision'],
                        'angle_output': ['Accuracy', 'MeanSquaredError', 'Recall', 'Precision'], 
                        })
        
        model.fit_generator(train_gen, 
                  steps_per_epoch=len(train_idx)//batch_size,
                  validation_data=val_gen, 
                  validation_steps=len(valid_idx)//batch_size,
                  epochs=150, 
                  callbacks=[callback, checkpoint, logger])
        
        # with open('logger_file.json', 'w') as f:
        #     json.dump(history, f) 
            
            
        model.save('multi_head_model_class_V3.h5')
    
    # def load_image(self, image_path):
        
    #         image = cv.imread(image_path)
    #         gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #         img = cv.medianBlur(gray_image,5)
    #         ret,threshold = cv.threshold(img,127,255,cv.THRESH_BINARY)
    #         contours, heirarchy = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #         #draw the obtained contour lines(or the set of coordinates forming a line) on the original image
    #         cv.drawContours(image, contours, -1, (0,0,255), 2)

    #         image = cv.resize(image, (244,244))  # Resize images
            
    #         return image
        
    # def preprocess(self, im_pth, label):
    #     image = self.load_image(im_pth)
    #     return image, label
    
    def get_backbone(self,inputs):
        x = layers.Rescaling(1./255, input_shape=input_shape)(inputs)
        x = pretrained_model(x)
        return x
    
    def get_angle_head(self,inputs):
        x = self.get_backbone(inputs)
        x = layers.GlobalAveragePooling2D()(x) 
        x = layers.Dense(640, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(17)(x)
        x = layers.Activation('softmax', name='angle_output')(x)
        return x
    
    def get_speed_head(self,inputs):
        x = self.get_backbone(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(1)(x)
        x = layers.Activation('sigmoid', name='speed_output')(x)
        return x
    
    # # Expand weights dimension to match new input channels
    # def multify_weights(self, kernel, out_channels):
    #     mean_1d = np.mean(kernel, axis=-2).reshape(kernel[:,:,-1:,:].shape)
    #     tiled = np.tile(mean_1d, (out_channels, 1))
    #     return(tiled)


    # # Loop through layers of both original model 
    # # and custom model and copy over weights 
    # # layer_modify refers to first convolutional layer
    # def copy_weights_tl(self, model_orig, custom_model, layer_modify):
    #     input_channel = 4
    #     layer_to_modify = [layer_modify]

    #     conf = custom_model.get_config()
    #     layer_names = [conf['layers'][x]['name'] for x in range(len(conf['layers']))]

    #     for layer in model_orig.layers:
    #         if layer.name in layer_names:
    #             if layer.get_weights() != []:
    #                 target_layer = custom_model.get_layer(layer.name)

    #                 if layer.name in layer_to_modify:    
    #                     kernels = layer.get_weights()[0]
    #                     biases  = layer.get_weights()[1]

    #                     kernels_extra_channel = np.concatenate((kernels,
    #                                                             self.multify_weights(kernels, input_channel - 3)),
    #                                                             axis=-2)
                                                                
    #                     target_layer.set_weights([kernels_extra_channel, biases])
    #                     target_layer.trainable = False

    #                 else:
    #                     target_layer.set_weights(layer.get_weights())
    #                     target_layer.trainable = False

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
        model_speed = models.load_model('model_speed.h5')
        model_angle = models.load_model('model_angle.h5')
        #model_regression = models.load_model('multi_head_model_reg.h5')
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
        
       # angle_pred, speed_pred = model_regression.predict(test_images)
        
        

        #loop through test data and produce predictions 
        for index, preds in enumerate(angle_predictions):
            
            # print(preds)
            #print(speed_predictions[index])
            # input()
            angle_pred = categories_angle[np.argmax(angle_predictions[index])]
            speed_pred = int(np.around(speed_predictions[index][0]))
            
            predictions['speed'].append(speed_pred)
            predictions['angle'].append(angle_pred)
            predictions['image_id'].append(re.split(r'[/.]', paths[index])[2])
            
        #save predictions and as a csv 
        df = pd.DataFrame(predictions)
        df.to_csv('outputs.csv', index = False)
        
if __name__ == "__main__":
    Predictor()
    
    




