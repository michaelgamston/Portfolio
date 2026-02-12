import tensorflow as tf

#speed_model = tf.keras.models.load_model('model_speed.h5')
angle_model = tf.keras.models.load_model('../../models/multi_head_model_class_V3.h5')


# converter_speed = tf.lite.TFLiteConverter.from_keras_model(speed_model) 
# converter_speed.optimizations = [tf.lite.Optimize.DEFAULT] 
# tflite_model_speed_quantized = converter_speed.convert()

converter_angle = tf.lite.TFLiteConverter.from_keras_model(angle_model) 
converter_angle.optimizations = [tf.lite.Optimize.DEFAULT] 
tflite_model_angle_quantized = converter_angle.convert()


# with open('speed_model_lite.tflite', 'wb') as f:     
#   f.write(tflite_model_speed_quantized)
  
with open('multi_head_model_class_V3.tflite', 'wb') as f:     
  f.write(tflite_model_angle_quantized)