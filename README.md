Tomato Leaf Disease Detection using CNN

This project trains a Convolutional Neural Network (CNN) to classify tomato leaf diseases using the dataset available on Kaggle. The dataset includes images of healthy and diseased tomato leaves categorized into different classes.

Dataset
The dataset is sourced from Kaggle: Tomato Leaf Dataset. It contains images categorized into train, validation, and test sets.

Prerequisites
Before running the project, ensure the following are installed:
Python 3.8 or higher
Required Python libraries:
TensorFlow
Keras
Kaggle
NumPy
Matplotlib
Install the dependencies using:
pip install tensorflow keras kaggle numpy matplotlib
Kaggle API setup:
Download your Kaggle API key from your Kaggle account.
Place the kaggle.json file in the ~/.kaggle/ directory on Linux/Mac or %USERPROFILE%\.kaggle\ on Windows.

Project Structure

├── tomato_leaf_cnn_model.h5   
├── script.py                  
├── README.md                  
├── dataset/                  
│   ├── train/                 
│   ├── val/                   
│   └── test/                

How to Run
Clone the repository or copy the script file.
Run the script to download the dataset and train the model:
python script.py

The script will:
Download the dataset directly from Kaggle.
Preprocess the images (rescaling and augmentation).
Train a CNN model on the training data.
Evaluate the model on the test data.
Save the trained model as tomato_leaf_cnn_model.h5.

CNN Architecture
The CNN consists of:
Three convolutional layers with ReLU activation and max pooling.
A fully connected dense layer with 128 units and a dropout of 50%.
An output layer with softmax activation for multi-class classification.

Results
The model's accuracy and loss are printed during training and evaluation. Test accuracy is displayed after evaluation.

Dataset link : https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf
