from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
from werkzeug.utils import secure_filename
import uuid
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure PostgreSQL database
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://username:password@localhost/brain_tumor_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # 'doctor' or 'patient'

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember='remember' in request.form)
            flash('Logged in successfully.', 'success')
            if user.role == 'doctor':
                return redirect(url_for('doctor_dashboard'))
            else:
                return redirect(url_for('patient_dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/doctor/dashboard', methods=['GET', 'POST'])
@login_required
def doctor_dashboard():
    if current_user.role != 'doctor':
        flash('Access denied.', 'danger')
        return redirect(url_for('home'))

    prediction = None
    gradcam_image = None
    damage_percent = None

    if request.method == 'POST':
        model_name = request.form['model_name']
        file = request.files.get('mri_image')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '_' + filename)
            file.save(filepath)

            # Load model based on selection
            model_path_map = {
                'Xception': 'Xception_trained_model.h5',
                'DenseNet121': 'DenseNet121_trained_model.h5',
                'MobileNet': 'MobileNet_trained_model.h5',
                'ResNet50': 'ResNet50_trained_model.h5'
            }
            model_path = model_path_map.get(model_name)
            if not model_path or not os.path.exists(model_path):
                flash('Selected model not found.', 'danger')
                return redirect(url_for('doctor_dashboard'))

            model = load_model(model_path)

            # Generate Grad-CAM and prediction
            prediction, gradcam_image_path, damage_percent = generate_prediction_and_gradcam(model, filepath, model_name)

            gradcam_image = gradcam_image_path

    return render_template('doctor_dashboard.html', prediction=prediction, gradcam_image=gradcam_image, damage_percent=damage_percent)

@app.route('/patient/dashboard', methods=['GET', 'POST'])
@login_required
def patient_dashboard():
    if current_user.role != 'patient':
        flash('Access denied.', 'danger')
        return redirect(url_for('home'))

    prediction = None
    gradcam_image = None
    damage_percent = None

    if request.method == 'POST':
        file = request.files.get('mri_image')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '_' + filename)
            file.save(filepath)

            # Default model for patients
            model_path = 'Xception_trained_model.h5'
            if not os.path.exists(model_path):
                flash('Model not found.', 'danger')
                return redirect(url_for('patient_dashboard'))

            model = load_model(model_path)

            # Generate Grad-CAM and prediction
            prediction, gradcam_image_path, damage_percent = generate_prediction_and_gradcam(model, filepath, 'Xception')

            gradcam_image = gradcam_image_path

    return render_template('patient_dashboard.html', prediction=prediction, gradcam_image=gradcam_image, damage_percent=damage_percent)

def generate_prediction_and_gradcam(model, img_path, model_name):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.xception.preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Adjust as per your classes
    prediction = class_labels[class_idx]

    # Calculate damage percent as confidence of predicted class
    damage_percent = round(float(preds[0][class_idx]) * 100, 2)

    # Generate Grad-CAM heatmap
    last_conv_layer_name = find_last_conv_layer(model)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=class_idx)

    # Load original image with OpenCV
    img_cv = cv2.imread(img_path)
    img_cv = cv2.resize(img_cv, (224, 224))

    heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

    # Save Grad-CAM image
    gradcam_filename = f'gradcam_outputs/{model_name}_{os.path.basename(img_path)}'
    os.makedirs('gradcam_outputs', exist_ok=True)
    cv2.imwrite(gradcam_filename, superimposed_img)

    return prediction, gradcam_filename, damage_percent

# Reuse find_last_conv_layer and make_gradcam_heatmap from previous code
def find_last_conv_layer(model):
    conv_layers = []

    def search_conv2d(layer):
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layers.append(layer)
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                search_conv2d(sublayer)

    search_conv2d(model)
    
    if not conv_layers:
        raise ValueError("No Conv2D layer found in the model.")
    
    last_conv = conv_layers[-1]
    return last_conv.name

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
    except ValueError:
        for layer in model.layers:
            if hasattr(layer, 'get_layer'):
                try:
                    nested_layer = layer.get_layer(last_conv_layer_name)
                    grad_model = tf.keras.models.Model(
                        [model.inputs],
                        [nested_layer.output, model.output]
                    )
                    break
                except ValueError:
                    continue
        else:
            raise

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

@app.route('/download_report')
@login_required
def download_report():
    # Generate PDF report for the last prediction (simplified example)
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.drawString(100, 750, f"Brain Tumor Classification Report for {current_user.username}")
    p.drawString(100, 730, "This is a sample report. Detailed results and images can be added here.")
    p.showPage()
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='report.pdf', mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)
