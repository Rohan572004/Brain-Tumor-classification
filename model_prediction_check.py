import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Load your trained model
model = tf.keras.models.load_model('trained_model.h5')

# Define class names (update as per your model)
class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))  # Adjust size to your model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

if __name__ == "__main__":
    # Example usage: provide path to image and true label
    test_image_path = r"D:\data science\Brain_tumor classifiaction\tumor_Testing\glioma\Te-gl_0010.jpg"
    true_label = "glioma"

    predicted_class, confidence = predict_image(test_image_path)
    print(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")
    print(f"True label: {true_label}")
    if predicted_class == true_label:
        print("Prediction is correct.")
    else:
        print("Prediction is incorrect.")
