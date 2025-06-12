# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import cv2
# import sys

# def get_img_array(img_path, size):
#     img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array / 255.0  # Normalize

# def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
#     # Access base MobileNet model inside the Sequential model
#     mobilenet_model = model.get_layer("mobilenet_1.00_128")
#     last_conv_layer = mobilenet_model.get_layer(last_conv_layer_name)
    
#     # Model to fetch last conv layer output
#     last_conv_model = tf.keras.Model(mobilenet_model.input, last_conv_layer.output)

#     # Model to apply classifier layers
#     classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
#     x = classifier_input
#     for layer_name in classifier_layer_names:
#         x = model.get_layer(layer_name)(x)
#     classifier_model = tf.keras.Model(classifier_input, x)

#     # Gradient computation
#     with tf.GradientTape() as tape:
#         conv_output = last_conv_model(img_array)
#         tape.watch(conv_output)
#         preds = classifier_model(conv_output)
#         class_channel = preds[:, tf.argmax(preds[0])]

#     grads = tape.gradient(class_channel, conv_output)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_output = conv_output[0]
#     heatmap = conv_output @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy()

# def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (128, 128))

#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap = np.uint8(255 * heatmap)

#     heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     superimposed_img = heatmap_color * alpha + img
#     cv2.imwrite(cam_path, superimposed_img)

#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Original Image")
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.title("Grad-CAM")
#     plt.imshow(cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB))
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     image_path = sys.argv[1]
#     model_path = sys.argv[2]

#     model = tf.keras.models.load_model(model_path)
#     img_array = get_img_array(image_path, size=(128, 128))

#     last_conv_layer_name = "conv_pw_13_relu"
#     classifier_layer_names = [
#         "global_average_pooling2d_9",
#         "dropout_16",
#         "dense_16",
#         "dropout_17",
#         "dense_17"
#     ]

#     heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names)
#     save_and_display_gradcam(image_path, heatmap)




# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import cv2
# import sys

# def get_img_array(img_path, size):
#     img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array / 255.0  # Normalize

# def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
#     mobilenet_model = model.get_layer("mobilenet_1.00_128")
#     last_conv_layer = mobilenet_model.get_layer(last_conv_layer_name)
    
#     last_conv_model = tf.keras.Model(mobilenet_model.input, last_conv_layer.output)

#     classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
#     x = classifier_input
#     for layer_name in classifier_layer_names:
#         x = model.get_layer(layer_name)(x)
#     classifier_model = tf.keras.Model(classifier_input, x)

#     with tf.GradientTape() as tape:
#         conv_output = last_conv_model(img_array)
#         tape.watch(conv_output)
#         preds = classifier_model(conv_output)
#         class_channel = preds[:, tf.argmax(preds[0])]

#     grads = tape.gradient(class_channel, conv_output)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_output = conv_output[0]
#     heatmap = conv_output @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy(), preds.numpy()[0]

# def save_and_display_gradcam(img_path, heatmap, predictions, class_names, cam_path="cam.jpg", alpha=0.4):
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (128, 128))

#     heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap_uint8 = np.uint8(255 * heatmap_resized)
#     heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

#     superimposed_img = heatmap_color * alpha + img
#     cv2.imwrite(cam_path, superimposed_img)

#     predicted_index = np.argmax(predictions)
#     predicted_label = class_names[predicted_index]
#     confidence = predictions[predicted_index] * 100

#     # Estimate damage: % of area with high activation
#     damage_threshold = 0.6
#     damage_percentage = np.sum(heatmap_resized > damage_threshold) / (128 * 128) * 100

#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Original Image")
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.title(f"Prediction: {predicted_label}\nConfidence: {confidence:.2f}%\nDamage Estimate: {damage_percentage:.2f}%")
#     plt.imshow(cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB))
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     image_path = sys.argv[1]
#     model_path = sys.argv[2]

#     class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

#     model = tf.keras.models.load_model(model_path)
#     img_array = get_img_array(image_path, size=(128, 128))

#     last_conv_layer_name = "conv_pw_13_relu"
#     classifier_layer_names = [
#         "global_average_pooling2d_9",
#         "dropout_16",
#         "dense_16",
#         "dropout_17",
#         "dense_17"
#     ]

#     heatmap, predictions = make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names)
#     save_and_display_gradcam(image_path, heatmap, predictions, class_names)



# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import cv2
# import sys
# import os
# from datetime import datetime

# def get_img_array(img_path, size):
#     img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array / 255.0  # Normalize

# def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
#     mobilenet_model = model.get_layer("mobilenet_1.00_128")
#     last_conv_layer = mobilenet_model.get_layer(last_conv_layer_name)
    
#     last_conv_model = tf.keras.Model(mobilenet_model.input, last_conv_layer.output)

#     classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
#     x = classifier_input
#     for layer_name in classifier_layer_names:
#         x = model.get_layer(layer_name)(x)
#     classifier_model = tf.keras.Model(classifier_input, x)

#     with tf.GradientTape() as tape:
#         conv_output = last_conv_model(img_array)
#         tape.watch(conv_output)
#         preds = classifier_model(conv_output)
#         class_channel = preds[:, tf.argmax(preds[0])]

#     grads = tape.gradient(class_channel, conv_output)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_output = conv_output[0]
#     heatmap = conv_output @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy(), preds.numpy()[0]

# def save_and_display_gradcam(img_path, heatmap, predictions, class_names, alpha=0.4):
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (128, 128))

#     # Resize heatmap using better interpolation
#     heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
#     heatmap_uint8 = np.uint8(255 * heatmap_resized)
#     heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

#     # Superimpose heatmap
#     superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

#     # Contour for high activation - using adaptive thresholding for better accuracy
#     gray_heatmap = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2GRAY)
    
#     # Calculate threshold dynamically based on the heatmap intensity distribution
#     # This makes the contours more adaptive to different images
#     threshold = int(np.percentile(gray_heatmap[gray_heatmap > 0], 70))  # Focus on top 30% of activations
    
#     # Apply threshold and find contours
#     _, binary = cv2.threshold(gray_heatmap, threshold, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Filter small contours to reduce noise (remove contours with small area)
#     min_contour_area = 20  # Minimum area threshold
#     significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
#     # Draw contours with a slightly thinner line for precision
#     cv2.drawContours(superimposed_img, significant_contours, -1, (0, 255, 0), 1)

#     # Prediction info (but we won't add it to the image)
#     predicted_index = np.argmax(predictions)
#     predicted_label = class_names[predicted_index]
#     confidence = predictions[predicted_index] * 100

#     # Damage estimate
#     damage_threshold = 0.6
#     damage_percentage = np.sum(heatmap_resized > damage_threshold) / (128 * 128) * 100

#     # Save to folder
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_dir = "gradcam_outputs"
#     os.makedirs(output_dir, exist_ok=True)
#     cam_path = os.path.join(output_dir, f"{predicted_label}_{timestamp}.jpg")
#     cv2.imwrite(cam_path, superimposed_img)

#     # Display
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Original Image")
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.title(f"Prediction: {predicted_label}\nConfidence: {confidence:.2f}%\nDamage Estimate: {damage_percentage:.2f}%")
#     plt.imshow(cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB))
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     image_path = sys.argv[1]
#     model_path = sys.argv[2]

#     class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

#     model = tf.keras.models.load_model(model_path)
#     img_array = get_img_array(image_path, size=(128, 128))

#     last_conv_layer_name = "conv_pw_13_relu"
#     classifier_layer_names = [
#         "global_average_pooling2d_9",
#         "dropout_16",
#         "dense_16",
#         "dropout_17",
#         "dense_17"
#     ]

#     heatmap, predictions = make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names)
#     save_and_display_gradcam(image_path, heatmap, predictions, class_names)



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import sys
import os
from datetime import datetime

def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalize

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    mobilenet_model = model.get_layer("mobilenet_1.00_128")
    last_conv_layer = mobilenet_model.get_layer(last_conv_layer_name)
    
    last_conv_model = tf.keras.Model(mobilenet_model.input, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_output = last_conv_model(img_array)
        tape.watch(conv_output)
        preds = classifier_model(conv_output)
        class_channel = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), preds.numpy()[0]

def save_and_display_gradcam(img_path, heatmap, predictions, class_names, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))

    # Resize heatmap using better interpolation
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Superimpose heatmap
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

    # Contour for high activation - using adaptive thresholding for better accuracy
    gray_heatmap = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2GRAY)
    
    # Calculate threshold dynamically based on the heatmap intensity distribution
    # This makes the contours more adaptive to different images
    threshold = int(np.percentile(gray_heatmap[gray_heatmap > 0], 70))  # Focus on top 30% of activations
    
    # Apply threshold and find contours
    _, binary = cv2.threshold(gray_heatmap, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small contours to reduce noise (remove contours with small area)
    min_contour_area = 20  # Minimum area threshold
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Draw contours with a slightly thinner line for precision
    cv2.drawContours(superimposed_img, significant_contours, -1, (0, 255, 0), 1)

    # Prediction info (but we won't add it to the image)
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    # Damage estimate
    damage_threshold = 0.6
    damage_percentage = np.sum(heatmap_resized > damage_threshold) / (128 * 128) * 100

    # Save to folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "gradcam_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the heatmap overlay image
    cam_path = os.path.join(output_dir, f"{predicted_label}_{timestamp}.jpg")
    cv2.imwrite(cam_path, superimposed_img)
    
    # Display
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Prediction: {predicted_label}\nConfidence: {confidence:.2f}%\nDamage Estimate: {damage_percentage:.2f}%")
    plt.imshow(cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    
    # Save the complete figure with both images and annotations
    fig_path = os.path.join(output_dir, f"{predicted_label}_{timestamp}_comparison.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print(f"Images saved to:\n1. {cam_path} (heatmap overlay)\n2. {fig_path} (comparison figure)")

if __name__ == "__main__":
    image_path = sys.argv[1]
    model_path = sys.argv[2]

    class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

    model = tf.keras.models.load_model(model_path)
    img_array = get_img_array(image_path, size=(128, 128))

    last_conv_layer_name = "conv_pw_13_relu"
    classifier_layer_names = [
        "global_average_pooling2d_9",
        "dropout_16",
        "dense_16",
        "dropout_17",
        "dense_17"
    ]

    heatmap, predictions = make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names)
    save_and_display_gradcam(image_path, heatmap, predictions, class_names)

