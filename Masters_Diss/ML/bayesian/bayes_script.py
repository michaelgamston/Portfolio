import keras
import tensorflow as tf
from keras.layers import LeakyReLU
import GPyOpt
from GPyOpt.methods import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('../../data/unlabelled_data.csv')
df.drop(columns=['Juniper'], inplace=True)

botanicals = df.iloc[:,25:277]


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        # Initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        # Initialize metrics to show loss during training
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
    def save_models(self):
        # Save the model to the specified filepath
        self.encoder.save("../../models/trad_arch_encoder.keras")
        self.decoder.save("../../models/trad_arch_decoder.keras")
        


    @property
    def metrics(self):
        # Return the list of metrics to track during training
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        
        with tf.GradientTape() as tape:
            #get the mean, log_var and  from the encoder
            mean,log_var, z = self.encoder(data) 
            #! add another layer here that adds the one hot encoded label of the botanicals to the decoder 
            #!info = labeler(data[1] , z)
            reconstruction = self.decoder(z)
            #calculate the reconstruction loss with binary crossentropy
            reconstruction_loss = keras.losses.binary_crossentropy(data, reconstruction)
                
            #calculate the KL divergence loss
            kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
            kl_loss = tf.reduce_mean(kl_loss, axis=1)
            
            #total loss is the sum of reconstruction loss and KL divergence
            total_loss = reconstruction_loss + kl_loss
            
        # Calculate gradients and apply them to the optimizer
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        
@keras.saving.register_keras_serializable()
class Sampling(keras.layers.Layer):
    """Uses (mean, log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon
    
def object_function(x):
    
        df = pd.read_csv('../../data/clustered_labels.csv')
        df.drop(columns=['Juniper'], inplace=True)
        
        botanicals = df.iloc[:,27:279]
        
        #split data into train and test
        train, test = train_test_split(botanicals, test_size=0.3, random_state=42)
        validation, test = train_test_split(test, test_size=0.67, random_state=42)
         
                
        h1_units = int(x[:, 0])
        latent_size = int(x[:, 3])
        num_of_layers = int(x[:, 4])
        
        #build input layer for encoder
        inputs_encoder = keras.Input(botanicals.columns.__len__()) 
        downsample1 = keras.layers.Dense(h1_units, activation=LeakyReLU())
        x_encoder = downsample1(inputs_encoder)
        
        #build hidden layers for encoder
        for layer in range(num_of_layers):
                
                if layer == 0:
                        h2_units = int(x[:, 1])
                        downsample2 = keras.layers.Dense(h2_units, activation=LeakyReLU())
                        x_encoder = downsample2(x_encoder)
                elif layer == 1:
                        h3_units = int(x[:, 2])
                        downsample3 = keras.layers.Dense(h3_units, activation=LeakyReLU())
                        x_encoder = downsample3(x_encoder)
                        
        #build latent space
        mean = keras.layers.Dense(latent_size, activation='linear')(x_encoder)
        log_var = keras.layers.Dense(latent_size, activation='linear')(x_encoder)
        z = Sampling()([mean, log_var])      
        
        #compile encoder model        
        encoder = keras.models.Model(inputs = inputs_encoder, outputs = [mean, log_var, z], name = "encoder")
            
                

        #build input layer for decoder
        latent_input = keras.Input(latent_size)
        
        upsampler1 = keras.layers.Dense(encoder.layers[num_of_layers+1].output_shape[-1], activation="relu")
        x_decoder = upsampler1(latent_input)
        
        #build hidden layers for decoder
        for layer in range(num_of_layers):
                if layer == 0:
                        upsampler2 = keras.layers.Dense(encoder.layers[num_of_layers].output_shape[-1], activation="relu")
                        x_decoder = upsampler2(x_decoder)
                elif layer == 1:
                        upsampler3 = keras.layers.Dense(encoder.layers[1].output_shape[-1], activation="relu")
                        x_decoder = upsampler3(x_decoder)
        
        #build output layer for decoder
        constructor = keras.layers.Dense(encoder.layers[0].output_shape[-1][1], activation="sigmoid")
        x_decoder = constructor(x_decoder)
        
        #compile decoder model
        decoder = keras.models.Model(inputs = latent_input, outputs = x_decoder, name = "decoder")
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.001)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('checkpoint.keras', monitor='loss', save_best_only=True)


        vae = VAE(encoder, decoder)
        vae.compile(optimizer=keras.optimizers.Adam())
        history = vae.fit(botanicals, epochs=50, batch_size=64,  callbacks = (checkpoint, callback))
        
        #evaluate model
        # predictions = vae.predict(test)
        # mse = mean_squared_error(test, predictions)
        
        
        return min(history.history["loss"])

# Setting the bounds of network parameter for the bayeyias optimizatio
bounds = [{'name': 'h1_units', 'type': 'discrete','domain': range(56, 560)},
            {'name': 'h2_units', 'type': 'discrete','domain': range(56, 560)},
            {'name': 'h3_units', 'type': 'discrete','domain': range(56, 560)},
            {'name': 'latent_size', 'type': 'discrete', 'domain': range(10, 60)},
            {'name': 'layers', 'type': 'discrete', 'domain': range(0,3)}]

# Creating the GPyOpt method using Bayesian Optimization
optimiser = GPyOpt.methods.BayesianOptimization(object_function, 
                                                   domain=bounds)

#Stop conditions
max_time  = None 
max_iter  = 100
tolerance = 1e-4


#Running the method
optimiser.run_optimization(max_iter = max_iter,
                            max_time = max_time,
                            eps = tolerance)

with open('bayes_opt.txt', 'a+') as report_file:
    report_file.write("\nNew report")
    report_file.write(f"\nValue of (x,y) that minimises the objective: \nh1_units:{optimiser.x_opt[0]}, \nh2_units:{optimiser.x_opt[1]}, \nh3_units:{optimiser.x_opt[2]} , \nlatent_size{optimiser.x_opt[3]}, \nnumber of hidden layers {optimiser.x_opt[4]}")
    report_file.write("\nMinimum value of the objective: "+str(optimiser.fx_opt))

# print("Value of (x,y) that minimises the objective:"+str(optimiser.x_opt))    
# print("Minimum value of the objective: "+str(optimiser.fx_opt))