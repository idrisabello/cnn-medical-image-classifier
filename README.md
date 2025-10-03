# cnn-medical-image-classifier


This project uses a Convolutional Neural Network (CNN) to classify medical images (e.g., X-rays or MRI scans) into different categories. It demonstrates a basic deep learning workflow — from dataset preparation to training, evaluation, and visualization of results — using TensorFlow/Keras.


# Project Structure

cnn-medical-image-classifier/
│
├── src/  Python source code for model and training
│ ├── train.py # Model training script
│ ├── evaluate.py # Model evaluation and prediction script
│
├── dataset/ # Folder where dataset should be placed
│ └── dataset_instructions.txt
│
├── outputs/ # Folder for generated plots, confusion matrix, etc.
│ ├── training_history.png
│ ├── confusion_matrix.png
│ └── sample_predictions.png
│
├── README.md # Project documentation
└── requirements.txt # Python dependencies


# Project Overview

The goal of this project is to build a basic CNN model capable of classifying medical images into their correct categories.  
While simple in structure, the project provides a strong foundation for beginners to understand how deep learning can be applied to real-world medical data.

Key Features:
-  CNN model built from scratch using Keras/TensorFlow  
-  Training and validation accuracy visualization  
-  Confusion matrix and sample prediction outputs  
-  Easily replaceable dataset for experimentation  


# Installation

To run this project, ensure you have **Python 3.11-3.10** installed. Then, in your terminal:


# 1. Clone the repository
git clone https://github.com/yourusername/cnn-medical-image-classifier.git

# 2. Navigate into the folder
cd cnn-medical-image-classifier

# 3. Install dependencies
pip install -r requirements.txt
Dataset
Due to licensing issues, the dataset is not included in this repository.
You can use any open-source medical image dataset such as:

Chest X-ray Pneumonia Dataset (Kaggle)

COVID-19 Radiography Database

Place your dataset into the dataset/ folder following these guidelines (see dataset_instructions.txt):


dataset/
    train/
        class_1/
            img1.png
            img2.png
            ...
        class_2/
            img3.png
            img4.png
            ...
    test/
        class_1/
        class_2/
For testing/demo, you can use 10–20 images per class to keep training light.

# How to Run
1. Train the Model
python src/train.py
This will train the CNN model and save training history plots to the outputs/ folder.

2. Evaluate & Generate Predictions
python src/evaluate_predict.ipynb
This will:
Generate a confusion matrix
Produce prediction results on test data
Save sample prediction outputs in outputs/

# Results
Example results generated after training:

Metric	Value
Training Accuracy	50%
Validation Accuracy	50%
Test Accuracy	 78.21%


# Future Improvements
Add more CNN layers to improve accuracy
Use transfer learning with pretrained models like ResNet or VGG
Implement Grad-CAM to visualize feature maps
Deploy the model via a simple Flask or Streamlit app


# Author
Idris Akande Bello
Robotics • AI • ML • Deep Learning • Embedded Systems
Speed Educational Consults, Nigeria
akandeidrisbello@gmail.com

