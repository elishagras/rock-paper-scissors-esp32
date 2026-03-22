import numpy as np
import tensorflow as tf

print("Loading model...")
# Load the trained Keras model saved by train_cnn.py
model = tf.keras.models.load_model("rps_model.h5")

print("Converting to TFLite...")
# Convert the Keras model to TensorFlow Lite format
# TFLite is a lightweight format required as an intermediate step
# before converting to the final tmdl format for the ESP32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to disk
# This file will be further converted by tflite2tmdl.py
with open("rps_model.tflite", "wb") as f:
    f.write(tflite_model)
print("Saved rps_model.tflite")
