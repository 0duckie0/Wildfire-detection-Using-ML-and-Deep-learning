:

🔥 Wildfire Detection Using Deep Learning
📌 Overview
This project leverages Machine Learning (ML) and Deep Learning techniques to detect wildfires from satellite imagery. A Convolutional Neural Network (CNN) is trained to classify images into two categories: wildfire and no wildfire. The model processes image data and predicts the presence of a wildfire, helping in early detection and prevention.

🚀 Features
Utilizes TensorFlow/Keras for deep learning.
Image preprocessing with ImageDataGenerator.
Binary classification model with CNN.
Trained on a dataset of wildfire and non-wildfire images.
Supports real-time image classification.
📂 Dataset
The dataset contains images categorized as:

Wildfire 🌲🔥
No Wildfire 🌲❌
Images are stored in directories:

swift
Copy
Edit
/kaggle/input/wildfire-prediction-dataset/train
/kaggle/input/wildfire-prediction-dataset/valid
/kaggle/input/wildfire-prediction-dataset/test
🏗️ Model Architecture
The CNN model consists of:

Conv2D layers for feature extraction
MaxPooling2D layers for downsampling
Dense layers for classification
Dropout to prevent overfitting
Sigmoid activation for binary classification
🛠️ Technologies Used
Python
TensorFlow/Keras
NumPy
Pandas
PIL (Python Imaging Library)
Tkinter (for GUI support, if needed)
📊 Training & Evaluation
Input images are resized to 64x64 pixels.
Data is augmented using ImageDataGenerator.
Loss function: Binary Crossentropy.
Optimizer: Adam.
Accuracy is used as the primary metric.
⚡ How to Use
Clone this repository:
bash
Copy
Edit
git clone https://github.com/yourusername/wildfire-detection.git
cd wildfire-detection
Install dependencies:
bash
Copy
Edit
pip install tensorflow numpy pandas pillow
Train the model:
python
Copy
Edit
python train.py
Use the trained model for predictions:
python
Copy
Edit
python predict.py --image test.jpg
📌 Future Enhancements
Improve model accuracy with Transfer Learning.
Deploy as a web application for real-time wildfire detection.
Integrate geospatial data for enhanced predictions.
📜 License
This project is open-source under the MIT License.
THE DATASET USED TO TRAIN THE MODEL : https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset

