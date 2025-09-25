from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np

name = ''
app = Flask(name)

# Chargement du module Keras
model = keras.models.load_model('mnist_model.h5')
@ app.route(' / predict', methods = ['POST'])
def predict():
    data = request.json
    # Verification des donnees
    if 'image' not in data:
        return jsonify({'error': 'Noimageprovided'}), 400

    image_data = np.array(data['image'])

    # Assurez-vous que l’image est au bon format (1, 784) et normalise
    image_data = image_data.reshape(1, 784)
    image_data = image_data.astype("float32") / 255.0
    prediction = model.predict(image_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return jsonify({'prediction': int(predicted_class),'probabilities': prediction.tolist()})

if name == 'main':
    app.run(host='0.0.0.0', port=5000)
