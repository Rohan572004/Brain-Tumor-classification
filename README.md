# Brain Tumor Classification Project

## Overview
This project is a brain tumor classification system that uses deep learning models to classify brain tumor types from medical images. It includes data preprocessing, model training, evaluation, and visualization components. The system also provides a web interface for users such as doctors and patients to interact with the model predictions.

## Features
- Classification of brain tumors into multiple categories (glioma, meningioma, pituitary, no tumor).
- Deep learning models used include DenseNet121, MobileNet, ResNet50, and Xception.
- Grad-CAM visualization for model interpretability.
- Web application with user authentication and dashboards for doctors and patients.
- Database integration for storing user and scan history data.
- Upload and prediction of brain MRI images through the web interface.

## Project Structure
- `app.py`: Main Flask application entry point.
- `brain_tumor_app.py`: Core application logic for brain tumor classification.
- `auth.py`: User authentication and authorization.
- `db_config.py`: Database configuration and setup.
- `init_db.py`: Database initialization script.
- `gradcam_visualizer.py`: Grad-CAM visualization utilities.
- `model_prediction_check.py`: Scripts for validating model predictions.
- `tumor_testing1.ipynb`: Jupyter notebook for testing and analysis.
- `templates/`: HTML templates for the web interface.
- `models_trained/`: Pretrained model files.
- `tumor_Training/` and `tumor_Testing/`: Dataset directories for training and testing images.
- `requirements.txt`: Python dependencies.

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd Brain_tumor_classification
   ```

2. Create and activate a Python virtual environment:
   ```
   python -m venv brain
   brain\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Initialize the database:
   ```
   python init_db.py
   ```

## Usage
1. Run the Flask application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:5000`.

3. Use the web interface to upload MRI images and get tumor classification results.

## Model Accuracy and Evaluation

The project uses multiple deep learning models trained on brain tumor MRI images. The models were evaluated on a test dataset with the following accuracy metrics:

| Model      | Accuracy | Precision | Recall | F1 Score |
|------------|----------|-----------|--------|----------|
| DenseNet121| 0.95     | 0.95      | 0.95   | 0.95     |
| ResNet50   | 0.93     | 0.93      | 0.93   | 0.93     |
| Xception   | 0.94     | 0.94      | 0.94   | 0.94     |
| MobileNet  | 0.92     | 0.92      | 0.92   | 0.92     |
| Ensemble   | 0.96     | 0.96      | 0.96   | 0.96     |

Confusion matrices and classification reports are generated for each model to assess precision, recall, and F1 scores. An ensemble model combining predictions from all models achieves the highest accuracy.

## Testing
- Use the Jupyter notebooks (`tumor_test.ipynb`, `tumor_testing1.ipynb`) for exploratory data analysis and testing.
- Run `model_prediction_check.py` to validate model predictions on test data.
- Test the web interface for image upload and prediction functionality.

## Contributing
Contributions are welcome. Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License.


