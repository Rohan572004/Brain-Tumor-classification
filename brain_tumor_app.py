# # Initial setup for extended version
# # Features to implement:
# # 1. Doctor/Patient login
# # 2. SQLite DB to store scan history
# # 3. Multiple image upload
# # 4. Export report as PDF
# # 5. Chat room for doctor-patient communication

# import streamlit as st
# st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")

# import tensorflow as tf
# import numpy as np
# import cv2
# from PIL import Image
# import sqlite3
# import os
# import io
# import base64
# from fpdf import FPDF

# # Initialize SQLite database
# conn = sqlite3.connect("scan_history.db", check_same_thread=False)
# c = conn.cursor()
# c.execute("""
# CREATE TABLE IF NOT EXISTS scans (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     username TEXT,
#     role TEXT,
#     prediction TEXT,
#     confidence REAL,
#     image BLOB
# )
# """)
# conn.commit()

# # Load model
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model("MobileNet_trained_model.h5")

# model = load_model()
# CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# # Grad-CAM Function
# def get_gradcam(image_array, model, last_conv_layer_name="Conv_1"):
#     img_tensor = np.expand_dims(image_array, axis=0)
#     grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_tensor)
#         class_idx = tf.argmax(predictions[0])
#         output = predictions[:, class_idx]
#     grads = tape.gradient(output, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]
#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     heatmap = heatmap.numpy()
#     heatmap = cv2.resize(heatmap, (image_array.shape[1], image_array.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     superimposed_img = cv2.addWeighted(cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)
#     return superimposed_img

# # PDF Export
# def export_pdf(report_text):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#     for line in report_text.splitlines():
#         pdf.cell(200, 10, txt=line, ln=True)
#     return pdf.output(dest='S').encode('latin1')

# # User Login System
# users = {"doctor1": ("doc123", "doctor"), "patient1": ("pat123", "patient")}

# # Chat system
# chat_log = []

# def login():
#     st.sidebar.header("Login")
#     username = st.sidebar.text_input("Username")
#     password = st.sidebar.text_input("Password", type="password")
#     if st.sidebar.button("Login"):
#         if username in users and users[username][0] == password:
#             st.session_state.logged_in = True
#             st.session_state.username = username
#             st.session_state.role = users[username][1]
#         else:
#             st.sidebar.error("Invalid credentials")

# def show_chat():
#     st.subheader("💬 Doctor-Patient Chat Room")
#     for entry in chat_log:
#         st.write(entry)
#     message = st.text_input("Type a message")
#     if st.button("Send"):
#         chat_log.append(f"{st.session_state.username}: {message}")

# # Main App
# if 'logged_in' not in st.session_state:
#     st.session_state.logged_in = False

# if not st.session_state.logged_in:
#     login()
# else:
#     st.sidebar.success(f"Logged in as {st.session_state.username} ({st.session_state.role})")
#     uploaded_files = st.sidebar.file_uploader("Upload MRI Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             image = Image.open(uploaded_file).convert('RGB')
#             st.image(image, caption=f"{uploaded_file.name}", width=300)
#             image_array = np.array(image.resize((224, 224))) / 255.0
#             prediction = model.predict(np.expand_dims(image_array, axis=0))[0]
#             predicted_class = CLASS_NAMES[np.argmax(prediction)]
#             confidence = np.max(prediction)
#             st.success(f"Prediction: {predicted_class}")
#             st.info(f"Confidence: {confidence:.2%}")
#             gradcam_img = get_gradcam(np.array(image.resize((224, 224))), model)
#             st.image(gradcam_img, caption="Grad-CAM Heatmap", channels="BGR")
#             report = f"User: {st.session_state.username}\nPrediction: {predicted_class}\nConfidence: {confidence:.2%}"
#             if st.button(f"Export PDF for {uploaded_file.name}"):
#                 pdf = export_pdf(report)
#                 st.download_button("Download Report PDF", data=pdf, file_name=f"report_{uploaded_file.name}.pdf")
#             if st.session_state.role == "doctor":
#                 # Store to DB
#                 c.execute("INSERT INTO scans (username, role, prediction, confidence, image) VALUES (?, ?, ?, ?, ?)",
#                           (st.session_state.username, st.session_state.role, predicted_class, float(confidence), uploaded_file.read()))
#                 conn.commit()
#     show_chat()







# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import os
# import sqlite3
# from PIL import Image
# import pandas as pd
# import time
# from reportlab.pdfgen import canvas
# import tempfile
# import base64

# # Database setup
# conn = sqlite3.connect("users.db", check_same_thread=False)
# c = conn.cursor()
# c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, role TEXT)''')
# c.execute('''CREATE TABLE IF NOT EXISTS history (username TEXT, filename TEXT, prediction TEXT, date TEXT)''')
# conn.commit()

# # Utility functions
# def authenticate(username, password, role):
#     c.execute("SELECT * FROM users WHERE username=? AND password=? AND role=?", (username, password, role))
#     return c.fetchone()

# def add_user(username, password, role):
#     try:
#         c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, password, role))
#         conn.commit()
#         return True
#     except sqlite3.IntegrityError:
#         return False

# def save_history(username, filename, prediction):
#     c.execute("INSERT INTO history (username, filename, prediction, date) VALUES (?, ?, ?, datetime('now'))", (username, filename, prediction))
#     conn.commit()

# def get_history(username):
#     c.execute("SELECT filename, prediction, date FROM history WHERE username=?", (username,))
#     return c.fetchall()

# def predict_image(model, image):
#     image = image.resize((224, 224))
#     img_array = tf.keras.utils.img_to_array(image)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     predictions = model.predict(img_array)
#     class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
#     return class_names[np.argmax(predictions)]

# def export_history_to_pdf(history, username):
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
#     c = canvas.Canvas(temp_file.name)
#     c.setFont("Helvetica", 12)
#     c.drawString(100, 800, f"Scan Report History for {username}")
#     y = 780
#     for filename, prediction, date in history:
#         c.drawString(100, y, f"File: {filename}, Prediction: {prediction}, Date: {date}")
#         y -= 20
#     c.save()
#     return temp_file.name

# def file_download_link(path, label):
#     with open(path, "rb") as f:
#         data = f.read()
#     b64 = base64.b64encode(data).decode()
#     href = f'<a href="data:application/octet-stream;base64,{b64}" download="report.pdf">{label}</a>'
#     return href

# def chat_interface(username):
#     st.subheader(f"Chat Room for {username}")
#     chat_file = f"chat_{username}.txt"
#     if os.path.exists(chat_file):
#         with open(chat_file, "r") as f:
#             chat_history = f.read()
#         st.text_area("Chat History", value=chat_history, height=300, disabled=True)
#     new_msg = st.text_input("Enter your message")
#     if st.button("Send"):
#         with open(chat_file, "a") as f:
#             f.write(f"{username}: {new_msg}\n")
#         st.experimental_rerun()

# # Main app
# def main():
#     st.set_page_config(layout="wide")
#     st.title("🧠 Brain Tumor Detection System")

#     menu = ["Login", "Sign Up"]
#     choice = st.sidebar.selectbox("Menu", menu)

#     if choice == "Login":
#         st.sidebar.subheader("Login Section")
#         username = st.sidebar.text_input("Username")
#         password = st.sidebar.text_input("Password", type='password')
#         role = st.sidebar.selectbox("Login as", ["doctor", "patient"])

#         if st.sidebar.button("Login"):
#             user = authenticate(username, password, role)
#             if user:
#                 st.success(f"Logged in as {username} ({role})")

#                 model = tf.keras.models.load_model("MobileNet_trained_model.h5")

#                 if role == "doctor":
#                     st.subheader("Upload MRI Scans")
#                     uploaded_files = st.file_uploader("Upload Multiple Images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
#                     if uploaded_files:
#                         for file in uploaded_files:
#                             image = Image.open(file)
#                             prediction = predict_image(model, image)
#                             st.image(image, caption=f"Prediction: {prediction}", use_column_width=True)
#                             save_history(username, file.name, prediction)

#                     st.subheader("Scan History")
#                     history = get_history(username)
#                     df = pd.DataFrame(history, columns=["Filename", "Prediction", "Date"])
#                     st.dataframe(df)

#                     if st.button("Export as PDF Report"):
#                         pdf_path = export_history_to_pdf(history, username)
#                         st.markdown(file_download_link(pdf_path, "📄 Download Report"), unsafe_allow_html=True)

#                     chat_interface(username)

#                 elif role == "patient":
#                     st.subheader("View Scan History")
#                     history = get_history(username)
#                     df = pd.DataFrame(history, columns=["Filename", "Prediction", "Date"])
#                     st.dataframe(df)

#                     if st.button("Export as PDF Report"):
#                         pdf_path = export_history_to_pdf(history, username)
#                         st.markdown(file_download_link(pdf_path, "📄 Download Report"), unsafe_allow_html=True)

#                     chat_interface(username)
#             else:
#                 st.error("Invalid credentials")

#     elif choice == "Sign Up":
#         st.sidebar.subheader("Create New Account")
#         new_user = st.sidebar.text_input("Username")
#         new_password = st.sidebar.text_input("Password", type='password')
#         role = st.sidebar.selectbox("Register as", ["doctor", "patient"])

#         if st.sidebar.button("Sign Up"):
#             success = add_user(new_user, new_password, role)
#             if success:
#                 st.success("Account created successfully")
#                 st.info("Go to Login to continue")
#             else:
#                 st.warning("Username already exists")

# if __name__ == '__main__':
#     main()





# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import os
# import sqlite3
# from PIL import Image
# import pandas as pd
# import time
# from reportlab.pdfgen import canvas
# import tempfile
# import base64
# import hashlib

# # --- Database setup ---
# conn = sqlite3.connect("users.db", check_same_thread=False)
# c = conn.cursor()
# c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, role TEXT)''')
# c.execute('''CREATE TABLE IF NOT EXISTS history (username TEXT, filename TEXT, prediction TEXT, date TEXT)''')
# conn.commit()

# # --- Utility functions ---

# def hash_password(password):
#     return hashlib.sha256(password.encode()).hexdigest()

# def authenticate(username, password, role):
#     hashed_pw = hash_password(password)
#     c.execute("SELECT * FROM users WHERE username=? AND password=? AND role=?", (username, hashed_pw, role))
#     return c.fetchone()

# def add_user(username, password, role):
#     try:
#         hashed_pw = hash_password(password)
#         c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, hashed_pw, role))
#         conn.commit()
#         return True
#     except sqlite3.IntegrityError:
#         return False

# def save_history(username, filename, prediction):
#     c.execute("INSERT INTO history (username, filename, prediction, date) VALUES (?, ?, ?, datetime('now'))", 
#               (username, filename, prediction))
#     conn.commit()

# def get_history(username):
#     c.execute("SELECT filename, prediction, date FROM history WHERE username=?", (username,))
#     return c.fetchall()

# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model("MobileNet_trained_model.h5")

# def predict_image(model, image):
#     if image.mode != "RGB":
#         image = image.convert("RGB")
#     image = image.resize((224, 224))
#     img_array = tf.keras.utils.img_to_array(image)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     predictions = model.predict(img_array)
#     class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
#     return class_names[np.argmax(predictions)]

# def export_history_to_pdf(history, username):
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
#     c = canvas.Canvas(temp_file.name)
#     c.setFont("Helvetica", 12)
#     c.drawString(100, 800, f"Scan Report History for {username}")
#     y = 780
#     for filename, prediction, date in history:
#         c.drawString(100, y, f"File: {filename}, Prediction: {prediction}, Date: {date}")
#         y -= 20
#         if y < 50:
#             c.showPage()
#             y = 800
#     c.save()
#     return temp_file.name

# def file_download_link(path, label):
#     with open(path, "rb") as f:
#         data = f.read()
#     b64 = base64.b64encode(data).decode()
#     href = f'<a href="data:application/octet-stream;base64,{b64}" download="report.pdf">{label}</a>'
#     return href

# def chat_interface(username):
#     st.subheader(f"Chat Room for {username}")
#     chat_file = f"chat_{username}.txt"
#     if os.path.exists(chat_file):
#         with open(chat_file, "r") as f:
#             chat_history = f.read()
#         st.text_area("Chat History", value=chat_history, height=300, disabled=True)
#     new_msg = st.text_input("Enter your message")
#     if new_msg and st.button("Send"):
#         with open(chat_file, "a") as f:
#             f.write(f"{username}: {new_msg}\n")
#         st.experimental_rerun()

# # --- Main App ---
# def main():
#     st.set_page_config(layout="wide")
#     st.title("🧠 Brain Tumor Detection System")

#     menu = ["Login", "Sign Up"]
#     choice = st.sidebar.selectbox("Menu", menu)

#     if choice == "Login":
#         st.sidebar.subheader("Login Section")
#         username = st.sidebar.text_input("Username")
#         password = st.sidebar.text_input("Password", type='password')
#         role = st.sidebar.selectbox("Login as", ["doctor", "patient"])

#         if st.sidebar.button("Login"):
#             user = authenticate(username, password, role)
#             if user:
#                 st.success(f"Logged in as {username} ({role})")

#                 model = load_model()

#                 if role == "doctor":
#                     st.subheader("Upload MRI Scans")
#                     uploaded_files = st.file_uploader("Upload Multiple Images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
#                     if uploaded_files:
#                         for file in uploaded_files:
#                             image = Image.open(file)
#                             filename = f"{int(time.time())}_{file.name}"
#                             prediction = predict_image(model, image)
#                             st.image(image, caption=f"Prediction: {prediction}", use_column_width=True)
#                             save_history(username, filename, prediction)

#                     st.subheader("Scan History")
#                     history = get_history(username)
#                     df = pd.DataFrame(history, columns=["Filename", "Prediction", "Date"])
#                     st.dataframe(df)

#                     if st.button("Export as PDF Report"):
#                         pdf_path = export_history_to_pdf(history, username)
#                         st.markdown(file_download_link(pdf_path, "📄 Download Report"), unsafe_allow_html=True)

#                     chat_interface(username)

#                 elif role == "patient":
#                     st.subheader("View Scan History")
#                     history = get_history(username)
#                     df = pd.DataFrame(history, columns=["Filename", "Prediction", "Date"])
#                     st.dataframe(df)

#                     if st.button("Export as PDF Report"):
#                         pdf_path = export_history_to_pdf(history, username)
#                         st.markdown(file_download_link(pdf_path, "📄 Download Report"), unsafe_allow_html=True)

#                     chat_interface(username)
#             else:
#                 st.error("Invalid credentials")

#     elif choice == "Sign Up":
#         st.sidebar.subheader("Create New Account")
#         new_user = st.sidebar.text_input("Username")
#         new_password = st.sidebar.text_input("Password", type='password')
#         role = st.sidebar.selectbox("Register as", ["doctor", "patient"])

#         if st.sidebar.button("Sign Up"):
#             success = add_user(new_user, new_password, role)
#             if success:
#                 st.success("Account created successfully")
#                 st.info("Go to Login to continue")
#             else:
#                 st.warning("Username already exists")

# if __name__ == '__main__':
#     main()










# import streamlit as st                                                  
# import os
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from PIL import Image

# # Set Streamlit page config (must be first Streamlit command)
# st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# st.title("Brain Tumor Classification Web App")

# # Folder path where models are saved
# MODEL_DIR = r"D:\data science\Brain_tumor classifiaction\models_trained"

# # Get list of .h5 model files in the folder
# try:
#     model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".h5")]
# except Exception as e:
#     st.error(f"Error accessing model directory: {e}")
#     model_files = []

# if not model_files:
#     st.error("No model files found in the models directory!")
#     st.stop()

# # Sidebar model selector
# selected_model_name = st.sidebar.selectbox("Select a Model", model_files)

# # Load the selected model
# model = None
# try:
#     model_path = os.path.join(MODEL_DIR, selected_model_name)
#     model = load_model(model_path)
#     st.sidebar.success(f"Loaded model: {selected_model_name}")
# except Exception as e:
#     st.sidebar.error(f"Failed to load model: {e}")

# # Show current model on main page
# st.write(f"### Current Model: {selected_model_name}")

# # File uploader to upload image
# uploaded_file = st.file_uploader("Choose a brain MRI image for prediction", type=["jpg", "jpeg", "png"])

# def preprocess_image(image_data, model):
#     """
#     Preprocess the image dynamically based on model input size.
#     """
#     # Get input shape of the model, e.g. (None, 128, 128, 3)
#     input_shape = model.input_shape  # tuple

#     # Extract height and width
#     # input_shape is usually like (None, height, width, channels)
#     height = input_shape[1]
#     width = input_shape[2]

#     img = Image.open(image_data).convert('RGB')
#     img = img.resize((width, height))
#     img_array = image.img_to_array(img)
#     img_array = img_array / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array




# if uploaded_file is not None and model is not None:
#     st.image(uploaded_file, caption='Uploaded MRI Image', use_container_width=True)

    
#     input_arr = preprocess_image(uploaded_file, model)
    
#     if st.button("Predict"):
#         try:
#             prediction = model.predict(input_arr)
#             predicted_class = np.argmax(prediction, axis=1)[0]
            
#             class_labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
            
#             if predicted_class < len(class_labels):
#                 st.success(f"Prediction: **{class_labels[predicted_class]}**")
#             else:
#                 st.error("Prediction class index out of range!")
#         except Exception as e:
#             st.error(f"Prediction failed: {e}")


# else:
#     if uploaded_file is None:
#         st.info("Please upload an image file to get started.")
#     if model is None:
#         st.warning("Please select a valid model from the sidebar.")                  working




# import streamlit as st
# import psycopg2
# import hashlib
# from datetime import datetime
# import numpy as np
# from PIL import Image
# import io
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import os

# # --- Database connection ---
# def get_connection():
#     return psycopg2.connect(
#         host="localhost",
#         database="brain_tumor_db",
#         user="your_username",
#         password="your_password"
#     )

# # --- Password hashing ---
# def hash_password(password):
#     return hashlib.sha256(password.encode()).hexdigest()

# # --- Check login credentials ---
# def check_login(username, password):
#     conn = get_connection()
#     cur = conn.cursor()
#     hashed_password = hash_password(password)
#     cur.execute("SELECT id, role FROM users WHERE username=%s AND password=%s", (username, hashed_password))
#     user = cur.fetchone()
#     cur.close()
#     conn.close()
#     if user:
#         return {"id": user[0], "role": user[1]}
#     return None

# # --- Load all models ---
# MODEL_DIR = r"D:\data science\Brain_tumor classifiaction\models_trained"
# model_files = {
#     "ResNet50": "resnet50_model.h5",
#     "Xception": "xception_model.h5",
#     "MobileNet": "mobilenet_model.h5",
#     "DenseNet": "densenet_model.h5"
# }
# models = {}
# for name, file in model_files.items():
#     models[name] = load_model(os.path.join(MODEL_DIR, file))

# # --- Prediction function ---
# def preprocess_image(image, target_size):
#     image = image.resize(target_size)
#     img_array = np.array(image)
#     if img_array.shape[2] == 4:
#         img_array = img_array[:, :, :3]
#     img_array = img_array / 255.0
#     return np.expand_dims(img_array, axis=0)

# def predict(image, model_name):
#     model = models[model_name]
#     target_size = (128, 128) if model_name != "Xception" else (224, 224)  # Adjust if needed
#     img = preprocess_image(image, target_size)
#     preds = model.predict(img)
#     class_idx = np.argmax(preds, axis=1)[0]
#     class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
#     return class_names[class_idx], float(np.max(preds))

# # --- Save scan report to DB ---
# def save_scan_report(user_id, tumor_class, probability, report, model_name):
#     conn = get_connection()
#     cur = conn.cursor()
#     cur.execute(
#         "INSERT INTO scans (user_id, tumor_class, probability, report, model_name) VALUES (%s, %s, %s, %s, %s)",
#         (user_id, tumor_class, probability, report, model_name)
#     )
#     conn.commit()
#     cur.close()
#     conn.close()

# # --- Fetch scans for a user ---
# def get_user_scans(user_id):
#     conn = get_connection()
#     cur = conn.cursor()
#     cur.execute("SELECT scan_date, tumor_class, probability, report, model_name FROM scans WHERE user_id=%s ORDER BY scan_date DESC", (user_id,))
#     scans = cur.fetchall()
#     cur.close()
#     conn.close()
#     return scans

# # --- Fetch all scans (for doctors) ---
# def get_all_scans():
#     conn = get_connection()
#     cur = conn.cursor()
#     cur.execute("""
#         SELECT users.username, scans.scan_date, scans.tumor_class, scans.probability, scans.report, scans.model_name
#         FROM scans JOIN users ON scans.user_id = users.id
#         ORDER BY scans.scan_date DESC
#     """)
#     scans = cur.fetchall()
#     cur.close()
#     conn.close()
#     return scans

# # --- Login UI ---
# def login():
#     st.title("Brain Tumor Classifier Login")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     if st.button("Login"):
#         user = check_login(username, password)
#         if user:
#             st.session_state['user'] = user
#             st.experimental_rerun()
#         else:
#             st.error("Invalid username or password")

# # --- Main app ---
# def main():
#     if 'user' not in st.session_state:
#         login()
#         return

#     user = st.session_state['user']
#     st.sidebar.write(f"Logged in as **{user['role']}** (User ID: {user['id']})")
#     if st.sidebar.button("Logout"):
#         del st.session_state['user']
#         st.experimental_rerun()

#     if user['role'] == 'patient':
#         st.header("Upload your brain MRI scan for tumor classification")
#         uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
#         model_name = st.selectbox("Select model for prediction", list(models.keys()))
#         if uploaded_file is not None:
#             image = Image.open(uploaded_file)
#             st.image(image, caption='Uploaded MRI Scan', use_container_width=True)
#             if st.button("Predict"):
#                 tumor_class, prob = predict(image, model_name)
#                 report = f"Prediction: {tumor_class} with confidence {prob:.2%}"
#                 st.success(report)
#                 # Save to DB
#                 save_scan_report(user['id'], tumor_class, prob, report, model_name)

#         st.subheader("Your previous scans:")
#         scans = get_user_scans(user['id'])
#         for scan in scans:
#             st.write(f"Date: {scan[0]}, Model: {scan[4]}, Tumor: {scan[1]}, Confidence: {scan[2]:.2%}")
#             st.write(f"Report: {scan[3]}")

#     elif user['role'] == 'doctor':
#         st.header("Doctor Dashboard - View patient scans")
#         scans = get_all_scans()
#         for scan in scans:
#             st.write(f"Patient: {scan[0]}, Date: {scan[1]}, Model: {scan[5]}, Tumor: {scan[2]}, Confidence: {scan[3]:.2%}")
#             st.write(f"Report: {scan[4]}")

# if __name__ == "__main__":
#     st.set_page_config(page_title="Brain Tumor Classification with Login", layout="centered")
#     main()




# import streamlit as st
# from auth import authenticate_user, register_user
# from db_config import execute_query
# import os
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import cv2

# # --------------------------- DB Save Prediction ---------------------------
# def save_prediction(user_id, model_name, prediction, gradcam_path):
#     query = """
#     INSERT INTO predictions (user_id, model_name, prediction, gradcam_image_path, prediction_date)
#     VALUES (%s, %s, %s, %s, NOW())
#     """
#     execute_query(query, (user_id, model_name, prediction, gradcam_path))

# # --------------------------- Model & Grad-CAM Utils -----------------------
# MODEL_DIR = r"D:\data science\Brain_tumor classifiaction\models_trained"

# def load_models():
#     models = {}
#     for model_name in ['resnet50', 'xception', 'mobilenet', 'densenet']:
#         path = os.path.join(MODEL_DIR, f"{model_name}.h5")
#         models[model_name] = tf.keras.models.load_model(path)
#     return models

# def preprocess_image(img, target_size):
#     img = img.resize(target_size)
#     img = np.array(img)
#     if img.shape[-1] == 4:
#         img = img[..., :3]
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)
#     return img

# def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
#     grad_model = tf.keras.models.Model(
#         [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
#     )
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         if pred_index is None:
#             pred_index = tf.argmax(predictions[0])
#         class_channel = predictions[:, pred_index]

#     grads = tape.gradient(class_channel, conv_outputs)

#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]
#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy()

# def save_and_overlay_gradcam(img_path, heatmap, output_path, alpha=0.4):
#     img = cv2.imread(img_path)
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     superimposed_img = heatmap_color * alpha + img
#     cv2.imwrite(output_path, superimposed_img)

# # --------------------------- Streamlit UI ----------------------------------
# st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False

# if not st.session_state.logged_in:
#     st.title("Login or Register")

#     option = st.selectbox("Select action", ["Login", "Register"])
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     role = None
#     if option == "Register":
#         role = st.selectbox("Role", ["doctor", "patient"])

#     if st.button(option):
#         if option == "Register":
#             if username and password and role:
#                 try:
#                     register_user(username, password, role)
#                     st.success("Registered successfully! Please login.")
#                 except Exception as e:
#                     st.error(f"Error: {e}")
#             else:
#                 st.warning("Fill all fields.")
#         elif option == "Login":
#             user = authenticate_user(username, password)
#             if user:
#                 st.session_state.logged_in = True
#                 st.session_state.user = user
#                 st.experimental_rerun()


#             else:
#                 st.error("Invalid username or password.")
# else:
#     st.title(f"Welcome {st.session_state.user['username']} ({st.session_state.user['role']})")

#     # Load models once and cache
#     @st.cache_resource
#     def get_models():
#         return load_models()
#     models = get_models()

#     uploaded_file = st.file_uploader("Upload MRI scan image", type=["png", "jpg", "jpeg"])

#     if uploaded_file is not None:
#         img = Image.open(uploaded_file)
#         st.image(img, caption="Uploaded Image", use_container_width=True)

#         # Select model to predict
#         model_name = st.selectbox("Choose model", list(models.keys()))
#         model = models[model_name]

#         # Set input size based on model - assuming 224x224 for all here; change if needed
#         input_size = (224, 224)
#         img_array = preprocess_image(img, input_size)

#         if st.button("Predict"):
#             prediction = model.predict(img_array)
#             pred_class_index = np.argmax(prediction[0])
#             classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']  # Update as per your classes
#             predicted_label = classes[pred_class_index]

#             st.success(f"Prediction: {predicted_label}")

#             # Grad-CAM
#             # You need to know the last conv layer name for each model:
#             last_conv_layers = {
#                 'resnet50': 'conv5_block3_out',
#                 'xception': 'block14_sepconv2_act',
#                 'mobilenet': 'Conv_1_relu',
#                 'densenet': 'conv5_block16_concat'
#             }
#             last_conv_layer = last_conv_layers.get(model_name, None)
#             if last_conv_layer:
#                 heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, pred_class_index)

#                 # Save Grad-CAM image
#                 gradcam_path = os.path.join("gradcam_images", f"{st.session_state.user['username']}_{model_name}_gradcam.jpg")
#                 os.makedirs("gradcam_images", exist_ok=True)

#                 # Save the overlay image
#                 img_path = os.path.join("temp", uploaded_file.name)
#                 os.makedirs("temp", exist_ok=True)
#                 img.save(img_path)

#                 save_and_overlay_gradcam(img_path, heatmap, gradcam_path)

#                 st.image(gradcam_path, caption="Grad-CAM")

#                 # Save prediction in DB
#                 save_prediction(
#                     user_id=st.session_state.user['id'],
#                     model_name=model_name,
#                     prediction=predicted_label,
#                     gradcam_path=gradcam_path
#                 )
#             else:
#                 st.warning("Grad-CAM not available for this model.")





# import streamlit as st
# from auth import authenticate_user, register_user
# from db_config import execute_query
# import os
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import cv2

# # --------------------------- DB Save Prediction ---------------------------
# def save_prediction(user_id, model_name, prediction, gradcam_path):
#     query = """
#     INSERT INTO predictions (user_id, model_name, prediction, gradcam_image_path, prediction_date)
#     VALUES (%s, %s, %s, %s, NOW())
#     """
#     execute_query(query, (user_id, model_name, prediction, gradcam_path))

# # --------------------------- Model & Grad-CAM Utils -----------------------
# MODEL_DIR = r"D:\data science\Brain_tumor classifiaction\models_trained"

# def load_models():
#     models = {
#         'ResNet50': tf.keras.models.load_model(os.path.join(MODEL_DIR, "ResNet50_trained_model.h5")),
#         'Xception': tf.keras.models.load_model(os.path.join(MODEL_DIR, "Xception_trained_model.h5")),
#         'MobileNet': tf.keras.models.load_model(os.path.join(MODEL_DIR, "MobileNet_trained_model.h5")),
#         'DenseNet121': tf.keras.models.load_model(os.path.join(MODEL_DIR, "DenseNet121_trained_model.h5")),
#     }
#     return models

# def preprocess_image(img, target_size):
#     img = img.resize(target_size)
#     img = np.array(img)
#     if img.shape[-1] == 4:
#         img = img[..., :3]
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)
#     return img

# def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
#     grad_model = tf.keras.models.Model(
#         [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
#     )
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         if pred_index is None:
#             pred_index = tf.argmax(predictions[0])
#         class_channel = predictions[:, pred_index]

#     grads = tape.gradient(class_channel, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]
#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy()

# def save_and_overlay_gradcam(img_path, heatmap, output_path, alpha=0.4):
#     img = cv2.imread(img_path)
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     superimposed_img = heatmap_color * alpha + img
#     cv2.imwrite(output_path, superimposed_img)

# # --------------------------- Streamlit UI ----------------------------------
# st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False
# if "page" not in st.session_state:
#     st.session_state.page = "login"  # default to login

# # --------------------------- Auth Pages -------------------------------------
# if not st.session_state.logged_in:
#     st.title("Login or Register")

#     if st.session_state.page == "login":
#         st.subheader("Login")
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")

#         if st.button("Login"):
#             user = authenticate_user(username, password)
#             if user:
#                 st.session_state.logged_in = True
#                 st.session_state.user = user
#                 st.rerun()
#             else:
#                 st.error("Invalid username or password.")

#         if st.button("Go to Register"):
#             st.session_state.page = "register"
#             st.rerun()

#     elif st.session_state.page == "register":
#         st.subheader("Register")
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")
#         confirm_password = st.text_input("Confirm Password", type="password")
#         role = st.selectbox("Role", ["doctor", "patient"])

#         if st.button("Register"):
#             if not username or not password or not confirm_password or not role:
#                 st.warning("Please fill all fields.")
#             elif password != confirm_password:
#                 st.warning("Passwords do not match.")
#             else:
#                 try:
#                     register_user(username, password, role)
#                     st.success("Registered successfully! Please login.")
#                     st.session_state.page = "login"
#                     st.rerun()
#                 except Exception as e:
#                     st.error(f"Error: {e}")

#         if st.button("Back to Login"):
#             st.session_state.page = "login"
#             st.rerun()

# # --------------------------- Main App (After Login) ------------------------
# else:
#     st.title(f"Welcome {st.session_state.user['username']} ({st.session_state.user['role']})")

#     @st.cache_resource
#     def get_models():
#         return load_models()
#     models = get_models()

#     uploaded_file = st.file_uploader("Upload MRI scan image", type=["png", "jpg", "jpeg"])

#     if uploaded_file is not None:
#         img = Image.open(uploaded_file)
#         st.image(img, caption="Uploaded Image", use_container_width=True)

#         model_name = st.selectbox("Choose model", list(models.keys()))
#         model = models[model_name]

#         input_size = (128, 128)
#         img_array = preprocess_image(img, input_size)

#         if st.button("Predict"):
#             prediction = model.predict(img_array)
#             pred_class_index = np.argmax(prediction[0])
#             classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
#             predicted_label = classes[pred_class_index]

#             st.success(f"Prediction: {predicted_label}")

#             last_conv_layers = {
#                 'ResNet50': 'conv5_block3_out',
#                 'Xception': 'block14_sepconv2_act',
#                 'MobileNet': 'Conv_1_relu',
#                 'DenseNet121': 'conv5_block16_concat'
#             }
#             last_conv_layer = last_conv_layers.get(model_name, None)

#             if last_conv_layer:
#                 heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, pred_class_index)

#                 gradcam_path = os.path.join("gradcam_images", f"{st.session_state.user['username']}_{model_name}_gradcam.jpg")
#                 os.makedirs("gradcam_images", exist_ok=True)

#                 img_path = os.path.join("temp", uploaded_file.name)
#                 os.makedirs("temp", exist_ok=True)
#                 img.save(img_path)

#                 save_and_overlay_gradcam(img_path, heatmap, gradcam_path)
#                 st.image(gradcam_path, caption="Grad-CAM")

#                 save_prediction(
#                     user_id=st.session_state.user['id'],
#                     model_name=model_name,
#                     prediction=predicted_label,
#                     gradcam_path=gradcam_path
#                 )
#             else:
#                 st.warning("Grad-CAM not available for this model.")







# import streamlit as st
# from auth import authenticate_user, register_user
# from db_config import execute_query
# import os
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import cv2

# # --------------------------- DB Save Prediction ---------------------------
# def save_prediction(user_id, model_name, prediction, gradcam_path):
#     query = """
#     INSERT INTO predictions (user_id, model_name, prediction, gradcam_image_path, prediction_date)
#     VALUES (%s, %s, %s, %s, NOW())
#     """
#     execute_query(query, (user_id, model_name, prediction, gradcam_path))

# # --------------------------- Model & Grad-CAM Utils -----------------------
# MODEL_DIR = r"D:\data science\Brain_tumor classifiaction\models_trained"

# def load_models():
#     models = {
#         'ResNet50': tf.keras.models.load_model(os.path.join(MODEL_DIR, "ResNet50_trained_model.h5")),
#         'Xception': tf.keras.models.load_model(os.path.join(MODEL_DIR, "Xception_trained_model.h5")),
#         'MobileNet': tf.keras.models.load_model(os.path.join(MODEL_DIR, "MobileNet_trained_model.h5")),
#         'DenseNet121': tf.keras.models.load_model(os.path.join(MODEL_DIR, "DenseNet121_trained_model.h5")),
#     }
#     return models

# def preprocess_image(img, target_size):
#     img = img.resize(target_size)
#     img = np.array(img)
#     if img.shape[-1] == 4:
#         img = img[..., :3]
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)
#     return img

# def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
#     grad_model = tf.keras.models.Model(
#         [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
#     )
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         if pred_index is None:
#             pred_index = tf.argmax(predictions[0])
#         class_channel = predictions[:, pred_index]

#     grads = tape.gradient(class_channel, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]
#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy()

# def save_and_overlay_gradcam(img_path, heatmap, output_path, alpha=0.4):
#     img = cv2.imread(img_path)
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap_uint8 = np.uint8(255 * heatmap)
#     heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
#     superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
#     cv2.imwrite(output_path, superimposed_img)

# # --------------------------- Streamlit UI ----------------------------------
# st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False
# if "page" not in st.session_state:
#     st.session_state.page = "login"  # default to login

# # --------------------------- Auth Pages -------------------------------------
# if not st.session_state.logged_in:
#     st.title("Login or Register")

#     if st.session_state.page == "login":
#         st.subheader("Login")
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")

#         if st.button("Login"):
#             user = authenticate_user(username, password)
#             if user:
#                 st.session_state.logged_in = True
#                 st.session_state.user = user
#                 st.experimental_rerun()
#             else:
#                 st.error("Invalid username or password.")

#         if st.button("Go to Register"):
#             st.session_state.page = "register"
#             st.experimental_rerun()

#     elif st.session_state.page == "register":
#         st.subheader("Register")
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")
#         confirm_password = st.text_input("Confirm Password", type="password")
#         role = st.selectbox("Role", ["doctor", "patient"])

#         if st.button("Register"):
#             if not username or not password or not confirm_password or not role:
#                 st.warning("Please fill all fields.")
#             elif password != confirm_password:
#                 st.warning("Passwords do not match.")
#             else:
#                 try:
#                     register_user(username, password, role)
#                     st.success("Registered successfully! Please login.")
#                     st.session_state.page = "login"
#                     st.experimental_rerun()
#                 except Exception as e:
#                     st.error(f"Error: {e}")

#         if st.button("Back to Login"):
#             st.session_state.page = "login"
#             st.experimental_rerun()

# # --------------------------- Main App (After Login) ------------------------
# else:
#     st.title(f"Welcome {st.session_state.user['username']} ({st.session_state.user['role']})")

#     @st.cache_resource
#     def get_models():
#         return load_models()
#     models = get_models()

#     uploaded_file = st.file_uploader("Upload MRI scan image", type=["png", "jpg", "jpeg"])

#     if uploaded_file is not None:
#         img = Image.open(uploaded_file)
#         st.image(img, caption="Uploaded Image", use_container_width=True)

#         model_name = st.selectbox("Choose model", list(models.keys()))
#         model = models[model_name]

#         input_size = (128, 128)
#         img_array = preprocess_image(img, input_size)

#         if st.button("Predict"):
#             prediction = model.predict(img_array)
#             pred_class_index = np.argmax(prediction[0])
#             classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
#             predicted_label = classes[pred_class_index]

#             st.success(f"Prediction: {predicted_label}")

#             last_conv_layers = {
#                 'ResNet50': 'conv5_block3_out',
#                 'Xception': 'block14_sepconv2_act',
#                 'MobileNet': 'conv_pw_13_relu',  # corrected layer name
#                 'DenseNet121': 'conv5_block16_concat'
#             }
#             last_conv_layer = last_conv_layers.get(model_name, None)

#             if last_conv_layer:
#                 heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, pred_class_index)

#                 gradcam_dir = "gradcam_images"
#                 os.makedirs(gradcam_dir, exist_ok=True)
#                 gradcam_path = os.path.join(gradcam_dir, f"{st.session_state.user['username']}_{model_name}_gradcam.jpg")

#                 temp_dir = "temp"
#                 os.makedirs(temp_dir, exist_ok=True)
#                 img_path = os.path.join(temp_dir, uploaded_file.name)
#                 img.save(img_path)

#                 save_and_overlay_gradcam(img_path, heatmap, gradcam_path, alpha=0.4)

#                 gradcam_img = cv2.imread(gradcam_path)
#                 gradcam_img = cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB)
#                 st.image(gradcam_img, caption="Grad-CAM Heatmap Overlay", use_container_width=True)

#                 save_prediction(
#                     user_id=st.session_state.user['id'],
#                     model_name=model_name,
#                     prediction=predicted_label,
#                     gradcam_path=gradcam_path
#                 )
#             else:
#                 st.warning("Grad-CAM not available for this model.")



import streamlit as st
from auth import authenticate_user, register_user
from db_config import execute_query
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# --------------------------- TF Config Fixes ---------------------------
tf.keras.backend.clear_session()
tf.config.run_functions_eagerly(True)

# --------------------------- DB Save Prediction ---------------------------
def save_prediction(user_id, model_name, prediction, gradcam_path):
    query = """
    INSERT INTO predictions (user_id, model_name, prediction, gradcam_image_path, prediction_date)
    VALUES (%s, %s, %s, %s, NOW())
    """
    execute_query(query, (user_id, model_name, prediction, gradcam_path))

# --------------------------- Model & Grad-CAM Utils -----------------------
MODEL_DIR = r"D:\data science\Brain_tumor classifiaction\models_trained"

def load_models():
    models = {
        'ResNet50': tf.keras.models.load_model(os.path.join(MODEL_DIR, "ResNet50_trained_model.h5")),
        'Xception': tf.keras.models.load_model(os.path.join(MODEL_DIR, "Xception_trained_model.h5")),
        'MobileNet': tf.keras.models.load_model(os.path.join(MODEL_DIR, "MobileNet_trained_model.h5")),
        'DenseNet121': tf.keras.models.load_model(os.path.join(MODEL_DIR, "DenseNet121_trained_model.h5")),
    }
    return models

def preprocess_image(img, target_size):
    img = img.resize(target_size)
    img = np.array(img)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    mobilenet_model = model.get_layer("mobilenet_1.00_128")
    last_conv_layer = mobilenet_model.get_layer(last_conv_layer_name)
    
    last_conv_model = tf.keras.Model(mobilenet_model.input, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    classifier_layer_names = [
        "global_average_pooling2d_9",
        "dropout_16",
        "dense_16",
        "dropout_17",
        "dense_17"
    ]
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_output = last_conv_model(img_array)
        tape.watch(conv_output)
        preds = classifier_model(conv_output)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), preds.numpy()[0]

def save_and_overlay_gradcam(img_path, heatmap, output_path, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

    # Contour for high activation - adaptive thresholding
    gray_heatmap = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2GRAY)
    threshold = int(np.percentile(gray_heatmap[gray_heatmap > 0], 70))
    _, binary = cv2.threshold(gray_heatmap, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 20
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    cv2.drawContours(superimposed_img, significant_contours, -1, (0, 255, 0), 1)

    cv2.imwrite(output_path, superimposed_img)

# --------------------------- Streamlit UI ----------------------------------
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"

# --------------------------- Auth Pages -------------------------------------
if not st.session_state.logged_in:
    st.title("Login or Register")

    if st.session_state.page == "login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = authenticate_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")

        if st.button("Go to Register"):
            st.session_state.page = "register"
            st.experimental_rerun()

    elif st.session_state.page == "register":
        st.subheader("Register")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        role = st.selectbox("Role", ["doctor", "patient"])

        if st.button("Register"):
            if not username or not password or not confirm_password or not role:
                st.warning("Please fill all fields.")
            elif password != confirm_password:
                st.warning("Passwords do not match.")
            else:
                try:
                    register_user(username, password, role)
                    st.success("Registered successfully! Please login.")
                    st.session_state.page = "login"
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.button("Back to Login"):
            st.session_state.page = "login"
            st.experimental_rerun()

# --------------------------- Main App (After Login) ------------------------
else:
    st.title(f"Welcome {st.session_state.user['username']} ({st.session_state.user['role']})")

    # Load models directly (no cache)
    models = load_models()

    uploaded_file = st.file_uploader("Upload MRI scan image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        model_name = st.selectbox("Choose model", list(models.keys()))
        model = models[model_name]

        input_size = (128, 128)
        img_array = preprocess_image(img, input_size)

        if st.button("Predict"):
            prediction = model.predict(img_array)
            pred_class_index = np.argmax(prediction[0])
            classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
            predicted_label = classes[pred_class_index]

            st.success(f"Prediction: {predicted_label}")

            # Dynamically find last Conv2D layer instead of hardcoded names
            def find_last_conv_layer(model):
                conv_layers = []

                def recursive_search(layer):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        conv_layers.append(layer)
                    elif hasattr(layer, 'layers'):
                        for sublayer in layer.layers:
                            recursive_search(sublayer)
                    elif hasattr(layer, 'layer'):
                        # For wrappers like TimeDistributed, Bidirectional, etc.
                        recursive_search(layer.layer)

                recursive_search(model)

                if conv_layers:
                    return conv_layers[-1].name
                else:
                    return None

            last_conv_layer = find_last_conv_layer(model)

            if last_conv_layer:
                heatmap, preds = make_gradcam_heatmap(img_array, model, last_conv_layer, pred_class_index)

                gradcam_dir = "gradcam_images"
                os.makedirs(gradcam_dir, exist_ok=True)
                gradcam_path = os.path.join(gradcam_dir, f"{st.session_state.user['username']}_{model_name}_gradcam.jpg")

                temp_dir = "temp"
                os.makedirs(temp_dir, exist_ok=True)
                img_path = os.path.join(temp_dir, uploaded_file.name)
                img.save(img_path)

                save_and_overlay_gradcam(img_path, heatmap, gradcam_path, alpha=0.4)

                gradcam_img = cv2.imread(gradcam_path)
                gradcam_img = cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB)
                st.image(gradcam_img, caption="Grad-CAM Heatmap Overlay", use_container_width=True)

                save_prediction(
                    user_id=st.session_state.user['id'],
                    model_name=model_name,
                    prediction=predicted_label,
                    gradcam_path=gradcam_path
                )
            else:
                st.warning("Grad-CAM not available for this model.")
