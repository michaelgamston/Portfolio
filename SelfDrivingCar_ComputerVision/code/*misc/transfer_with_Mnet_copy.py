import numpy as np
import tensorflow as tf
import keras
assert float(tf.__version__[:3]) >= 2.03
from keras.callbacks import EarlyStopping
import os
#import matplotlib.pyplot as plt

track_dir = '../combined'

IMAGE_SIZE = 224
BATCH_SIZE = 64

datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)

train_generator = datagen.flow_from_directory(
    track_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='training')

val_generator = datagen.flow_from_directory(
    track_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation')

image_batch, label_batch = next(val_generator)
image_batch.shape, label_batch.shape

# save the class labels to a text file:

print (train_generator.class_indices)

labels = '\n'.join(sorted(train_generator.class_indices.keys()))

with open('track_labels.txt', 'w') as f:
  f.write(labels)

# Build the model
# Create the base model

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Create the base model from the pre-trained MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False,
                                              weights='imagenet')
base_model.trainable = False

 

 
 
 # Add a classification head

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(units=34, activation='softmax')
])

# Configure the model


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.summary()

# print('Number of trainable weights = {}'.format(len(model.trainable_weights)))


# Train the model



early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=50,
                    validation_data=val_generator,
                    validation_steps=len(val_generator),
                    callbacks=[early_stopping])



# history = model.fit(train_generator,
#                     steps_per_epoch=len(train_generator),
#                     epochs=10,
#                     validation_data=val_generator,
#                     validation_steps=len(val_generator))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']



# review the learning curves


# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.ylabel('Accuracy')
# plt.ylim([min(plt.ylim()),1])
# plt.title('Training and Validation Accuracy')

# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross Entropy')
# plt.ylim([0,2.0])
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.show()

# Fine tune the base model
# Un-freeze more layers

# print("Number of layers in the base model: ", len(base_model.layers))




base_model.trainable = True
fine_tune_at = 50

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False


  model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history_fine = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=50,
                    validation_data=val_generator,
                    validation_steps=len(val_generator),
                    callbacks=[early_stopping])


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('mobilenet_v2_1.0_224.tflite', 'wb') as f:
  f.write(tflite_model)

  # A generator that provides a representative dataset
def representative_data_gen():
  dataset_list = tf.data.Dataset.list_files(track_dir + '/*/*')
  for i in range(100):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

with open('mobilenet_v2_1.0_224_quant.tflite', 'wb') as f:
  f.write(tflite_model)

  # A generator that provides a representative dataset
def representative_data_gen():
  dataset_list = tf.data.Dataset.list_files(track_dir + '/*/*')
  for i in range(100):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

with open('mobilenet_v2_1.0_224_quant.tflite', 'wb') as f:
  f.write(tflite_model)



batch_images, batch_labels = next(val_generator)

logits = model(batch_images)
prediction = np.argmax(logits, axis=1)
truth = np.argmax(batch_labels, axis=1)

keras_accuracy = tf.keras.metrics.Accuracy()
keras_accuracy(prediction, truth)

print("Raw model accuracy: {:.3%}".format(keras_accuracy.result()))

  

def set_input_tensor(interpreter, input):
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    # Inputs for the TFLite model must be uint8, so we quantize our input data.
    # NOTE: This step is necessary only because we're receiving input data from
    # ImageDataGenerator, which rescaled all image data to float [0,1]. When using
    # bitmap inputs, they're already uint8 [0,255] so this can be replaced with:
    #   input_tensor[:, :] = input
    scale, zero_point = input_details['quantization']
    input_tensor[:, :] = np.uint8(input / scale + zero_point)

def classify_image(interpreter, input):
  set_input_tensor(interpreter, input)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = interpreter.get_tensor(output_details['index'])
  # Outputs from the TFLite model are uint8, so we dequantize the results:
  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)
  top_1 = np.argmax(output)
  return top_1

interpreter = tf.lite.Interpreter('mobilenet_v2_1.0_224_quant.tflite')
interpreter.allocate_tensors()

# Collect all inference predictions in a list
batch_prediction = []
batch_truth = np.argmax(batch_labels, axis=1)

for i in range(len(batch_images)):
  prediction = classify_image(interpreter, batch_images[i])
  batch_prediction.append(prediction)

# Compare all predictions to the ground truth
tflite_accuracy = tf.keras.metrics.Accuracy()
tflite_accuracy(batch_prediction, batch_truth)
print("Quant TF Lite accuracy: {:.3%}".format(tflite_accuracy.result()))


# ! curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# ! echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

# ! sudo apt-get update

# ! sudo apt-get install edgetpu-compiler	

# ! edgetpu_compiler mobilenet_v2_1.0_224_quant.tflite



# import subprocess

# # Download the GPG key and save it directly to /etc/apt/trusted.gpg.d/
# result = subprocess.run(["curl", "-s", "https://packages.cloud.google.com/apt/doc/apt-key.gpg"], capture_output=True)

# if result.returncode == 0:  # Check if curl was successful
#     with open("/tmp/google.gpg", "wb") as f:
#         f.write(result.stdout)  # Save the key temporarily
#     subprocess.run(["sudo", "mv", "/tmp/google.gpg", "/etc/apt/trusted.gpg.d/google.gpg"])  # Move it to trusted keys directory
# else:
#     print("Failed to download GPG key:", result.stderr.decode())


# # Add GPG key
# # subprocess.run(["curl", "https://packages.cloud.google.com/apt/doc/apt-key.gpg"], stdout=subprocess.PIPE)
# # subprocess.run(["sudo", "apt-key", "add", "-"], input=subprocess.PIPE)

# # Add repository
# subprocess.run(["echo", "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main"], stdout=subprocess.PIPE)
# subprocess.run(["sudo", "tee", "/etc/apt/sources.list.d/coral-edgetpu.list"], input=subprocess.PIPE)

# # Update package lists
# subprocess.run(["sudo", "apt-get", "update"])

# # Install Edge TPU compiler
# subprocess.run(["sudo", "apt-get", "install", "-y", "edgetpu-compiler"])

# # Compile model
# subprocess.run(["edgetpu_compiler", "mobilenet_v2_1.0_224_quant.tflite"])




