#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import  EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train_data = "D:\data science\Brain_tumor classifiaction\tumor_Training"
test_data = "D:\data science\Brain_tumor classifiaction\tumor_Testing"


# In[3]:


import os

# Count images in training and testing directories
def count_images(data_path):
    total_images = 0
    print(f"Counting images in: {data_path}")
    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        if os.path.isdir(class_path):
            num_images = len(os.listdir(class_path))
            total_images += num_images
            print(f"{class_name}: {num_images} images")
    print(f"Total: {total_images} images\n")
    return total_images

# Count training and testing images
train_total = count_images(r"D:\data science\Brain_tumor classifiaction\tumor_Training")
test_total = count_images(r"D:\data science\Brain_tumor classifiaction\tumor_Testing")


# In[4]:


# class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']

# # Plot sample images from each class
# plt.figure(figsize=(12, 6))
# for i, class_name in enumerate(class_names):
#     class_path = os.path.join(train_data, class_name)
#     img_name = os.listdir(class_path)[0]  # Pick the first image in the folder
#     img_path = os.path.join(class_path, img_name)
#     img = Image.open(img_path)

#     plt.subplot(1, 4, i+1)
#     plt.imshow(img, cmap='gray')
#     plt.title(class_name.replace('_', ' ').title())
#     plt.axis('off')

# plt.tight_layout()
# plt.show()

import os
import matplotlib.pyplot as plt
from PIL import Image

train_data = r"D:\data science\Brain_tumor classifiaction\tumor_Training"

class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']

# Plot sample images from each class
plt.figure(figsize=(12, 6))
for i, class_name in enumerate(class_names):
    class_path = os.path.join(train_data, class_name)
    # Filter image files only
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        continue
    img_name = image_files[0]  # Pick the first image in the folder
    img_path = os.path.join(class_path, img_name)
    img = Image.open(img_path)

    plt.subplot(1, 4, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(class_name.replace('_', ' ').title())
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[ ]:


# Define image size and batch size (also added training Parameters later)
image_size = (128, 128)
batch_size = 32
num_classes = 4
input_shape = (128, 128, 3)
epochs = 60
learning_rate = 0.001


# In[30]:


# Data Augmentation & Preprocessing for Training
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.15,
#     zoom_range=0.15,
#     horizontal_flip=True,
#     validation_split=0.2,
#     rotation_range=15,
#     fill_mode='nearest',
# )

# enhanced data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2,
)


# In[7]:


# Create training dataset
train_generator = train_datagen.flow_from_directory(
    train_data,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=42 
)


# In[8]:


# Create validation dataset (without data augmentation)
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

validation_generator = validation_datagen.flow_from_directory(
    train_data,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle= False,
    seed= 42
)


# Let's use Transfer learning model to train them 

# In[9]:


num_classes = 4
input_shape = (128, 128, 3)
epoch = 60
lr = 0.001


# In[10]:


# Model
def createModel(base_model):

    # Add layer
    model = Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# In[11]:


def evaluate(model):
    y_pred = model.predict(validation_generator).argmax(axis = 1)
    y_true = validation_generator.classes

    class_labels = list(validation_generator.class_indices.keys())

    conf_mat = confusion_matrix(y_true, y_pred)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    display(HTML('<hr>'))

    # Classification report
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_labels))


# In[12]:


def historyPlot(history):
    # Plot training history
    plt.figure(figsize=(12, 6))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# In[13]:


# Implement early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
callbacks = [early_stopping, reduce_lr]


# In[14]:


def trainModel(model):
    
    # Create and compile the model
    model = createModel(model)
    model.summary()
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=epoch,
        validation_data=validation_generator,
        callbacks = callbacks,
        verbose = 0
    )

    return model, history


# In[23]:


from tensorflow.keras.applications import DenseNet121, ResNet50, Xception, MobileNet

trained_models = []

# Define base models
models_to_train = [
    ('DenseNet121', DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)),
    ('ResNet50', ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)),
    ('Xception', Xception(weights='imagenet', include_top=False, input_shape=input_shape)),
    ('MobileNet', MobileNet(weights='imagenet', include_top=False, input_shape=input_shape))
]

# Train the models
for name, base_model in models_to_train:
    display(HTML(f'<h2>{name}</h2>'))
    display(HTML('<hr>'))
    model, history = trainModel(base_model)
    # Save the trained model
    model.save(f'{name}_trained_model.h5')
    display(HTML('<hr>'))
    historyPlot(history)
    display(HTML('<hr>'))
    evaluate(model)
    display(HTML('<hr>'))
    # Store the trained model and name
    trained_models.append((name, model))


# In[24]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_data = r"D:\\data science\\Brain_tumor classifiaction\\tumor_Testing"
image_size = (224, 224)  # example target size, adjust as needed

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data,
    target_size=image_size,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)


# In[25]:


def evaluate_on_test(model, model_name):
    y_true = test_generator.classes
    y_pred_probs = model.predict(test_generator, verbose=0)
    y_pred = y_pred_probs.argmax(axis=1)

    class_labels = list(test_generator.class_indices.keys())

    conf_mat = confusion_matrix(y_true, y_pred)

    display(HTML(f'<h2>{model_name}</h2>'))
    display(HTML('<hr>'))

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification report
    display(HTML('<hr>'))
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_labels))
    display(HTML('<hr>'))
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1
    }


# In[26]:


results = []

for name, model in trained_models:
    result = evaluate_on_test(model, name)
    results.append(result)

# Convert to DataFrame
results_df = pd.DataFrame(results)
display(HTML(f'<h2>Comparison between Models</h2>'))
display(HTML('<hr>'))
results_df


# In[28]:


import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load your trained model (update the path to your saved model)
model = tf.keras.models.load_model("D:\data science\Brain_tumor classifiaction\Xception_trained_model.h5")

# Define class names (update as per your model)
class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))  # Adjust size as per your model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def upload_and_predict():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return

    try:
        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]

        # Display image
        img = Image.open(file_path)
        img.thumbnail((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Display prediction
        result_label.config(text=f"Predicted Tumor Type: {predicted_class}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image:\n{e}")

# Create main window
root = tk.Tk()
root.title("Brain Tumor Detection")
root.geometry("400x400")

# Upload button
upload_btn = tk.Button(root, text="Upload MRI Scan", command=upload_and_predict)
upload_btn.pack(pady=20)

# Image display label
image_label = tk.Label(root)
image_label.pack()

# Result label
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=20)

root.mainloop()


# In[20]:


import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Load your trained model
model = tf.keras.models.load_model('D:\data science\Brain_tumor classifiaction\Xception_trained_model.h5')

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
    test_image_path = r"D:\data science\Brain_tumor classifiaction\tumor_Testing\meningioma\Te-me_0010.jpg"
    true_label = "glioma"

    predicted_class, confidence = predict_image(test_image_path)
    print(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")
    print(f"True label: {true_label}")
    if predicted_class == true_label:
        print("Prediction is correct.")
    else:
        print("Prediction is incorrect.")


# In[32]:


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')

# Paths
train_data = r"D:\data science\Brain_tumor classifiaction\tumor_Training"
test_data = r"D:\data science\Brain_tumor classifiaction\tumor_Testing"

# Parameters
image_size = (128, 128)
batch_size = 32
num_classes = 4
input_shape = (128, 128, 3)
epochs = 60
learning_rate = 0.001

# Enhanced Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    train_data,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=42
)

validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
validation_generator = validation_datagen.flow_from_directory(
    train_data,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
callbacks = [early_stopping, reduce_lr]

# Model creation with fine-tuning option
def create_model(base_model, fine_tune_at=None):
    base_model.trainable = True
    if fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True
    else:
        base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Training function
def train_model(base_model, fine_tune_at=None):
    model = create_model(base_model, fine_tune_at)
    model.summary()
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    return model, history

# Import pretrained models
from tensorflow.keras.applications import DenseNet121, ResNet50, Xception, MobileNet

models_to_train = [
    ('DenseNet121', DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)),
    ('ResNet50', ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)),
    ('Xception', Xception(weights='imagenet', include_top=False, input_shape=input_shape)),
    ('MobileNet', MobileNet(weights='imagenet', include_top=False, input_shape=input_shape))
]

trained_models = []
for name, base_model in models_to_train:
    display(HTML(f'<h2>{name}</h2>'))
    display(HTML('<hr>'))
    # Fine-tune last 20 layers
    model, history = train_model(base_model, fine_tune_at=len(base_model.layers)-20)
    model.save(f'{name}_trained_model.h5')
    display(HTML('<hr>'))
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    display(HTML('<hr>'))
    # Evaluate on validation set
    y_pred = model.predict(validation_generator).argmax(axis=1)
    y_true = validation_generator.classes
    class_labels = list(validation_generator.class_indices.keys())
    conf_mat = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    display(HTML('<hr>'))
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_labels))
    display(HTML('<hr>'))
    trained_models.append((name, model))

# Test data generator with consistent image size
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data,
    target_size=image_size,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Evaluate on test set
def evaluate_on_test(model, model_name):
    y_true = test_generator.classes
    y_pred_probs = model.predict(test_generator, verbose=0)
    y_pred = y_pred_probs.argmax(axis=1)
    class_labels = list(test_generator.class_indices.keys())
    conf_mat = confusion_matrix(y_true, y_pred)
    display(HTML(f'<h2>{model_name}</h2>'))
    display(HTML('<hr>'))
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    display(HTML('<hr>'))
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_labels))
    display(HTML('<hr>'))
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1
    }

results = []
for name, model in trained_models:
    result = evaluate_on_test(model, name)
    results.append(result)

# Ensemble prediction by averaging probabilities
def ensemble_predict(models, generator):
    y_preds = [model.predict(generator, verbose=0) for _, model in models]
    avg_preds = np.mean(y_preds, axis=0)
    y_pred = avg_preds.argmax(axis=1)
    y_true = generator.classes
    class_labels = list(generator.class_indices.keys())
    conf_mat = confusion_matrix(y_true, y_pred)
    display(HTML(f'<h2>Ensemble Model</h2>'))
    display(HTML('<hr>'))
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    display(HTML('<hr>'))
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_labels))
    display(HTML('<hr>'))
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return {
        'Model': 'Ensemble',
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1
    }

ensemble_result = ensemble_predict(trained_models, test_generator)
results.append(ensemble_result)

results_df = pd.DataFrame(results)
display(HTML(f'<h2>Comparison between Models</h2>'))
display(HTML('<hr>'))
display(results_df)

