from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('models/xception_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    # Process the image
    image = Image.open(io.BytesIO(file.read()))
    image = image.resize((299, 299))  # Resize according to your model's input size
    image_array = np.array(image) / 255.0  # Normalize the image if needed
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)
    result = {'prediction': int(prediction[0][0] > 0.5)}  # Adjust threshold based on your model

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
