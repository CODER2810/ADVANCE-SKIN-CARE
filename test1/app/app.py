from flask import Flask, request, render_template
import joblib
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
svm_model = joblib.load('models/svm_model.pkl')
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Define class labels
class_labels = [
    'Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis',
    'Dermatofibroma', 'Melanocytic nevus', 'Melanoma',
    'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis', 'Vascular lesion'
]

# Feature extraction function
def extract_features(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    features = vgg16.predict(img_array)
    return features.flatten()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400
    if file:
        image_path = os.path.join('app/static/images', file.filename)
        file.save(image_path)
        features = extract_features(image_path)
        prediction = svm_model.predict([features])
        predicted_class = class_labels[prediction[0]]
        return render_template('result.html', image_path=image_path, prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)