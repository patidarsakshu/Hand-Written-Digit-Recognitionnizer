# Hand-Written-Digit-Recognitionnizer

Introduction

This project involves creating a 5-layer Sequential Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset. The model is built with the Keras API (using a TensorFlow backend), known for its user-friendly and intuitive interface. The project focuses on data preparation, model definition, training, evaluation, and prediction.
1. Data Preparation
1.1 Load Data
The first step involves loading the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits (0-9).

1.2 Check for Null and Missing Values
Before processing, it’s crucial to check for any null or missing values to ensure data integrity and prevent errors during training.

1.3 Normalization
Normalization scales the pixel values (0-255) to the range 0-1. This step is essential for faster convergence of the CNN during training.
1.4 Reshape
The data is reshaped to fit the input shape required by the CNN. For MNIST, each image is 28x28 pixels, and reshaping ensures that the model receives the data in the correct format.
1.5 Label Encoding
The labels (digits 0-9) are one-hot encoded to facilitate multi-class classification.
1.6 Split Training and Validation Set
The data is split into training and validation sets to evaluate the model's performance on unseen data during training.


2. Convolutional Neural Network (CNN)
2.1 Define the Model
The CNN model is defined with 5 layers, including convolutional layers, pooling layers, and dense (fully connected) layers. Each layer has specific functions, such as feature extraction and classification.
2.2 Set the Optimizer and Annealer
An optimizer (like Adam) is chosen to minimize the loss function, and an annealer (learning rate scheduler) is set to adjust the learning rate during training for better convergence.
2.3 Data Augmentation
Data augmentation techniques (like rotation, zoom, and shift) are applied to the training data to increase the diversity and robustness of the model.


3. Evaluate the Model
3.1 Training and Validation Curves
Training and validation curves (accuracy and loss) are plotted to monitor the model's performance and detect overfitting or underfitting during training.
3.2 Confusion Matrix
A confusion matrix is generated to visualize the model's performance in terms of true positives, true negatives, false positives, and false negatives.

4. Prediction and Submission
4.1 Predict and Submit Results
The trained model is used to predict the labels of the test dataset. The predictions are then formatted according to the required submission format for evaluation.
Results

The CNN achieved an impressive accuracy of 99.671% on the validation set, trained in approximately 2 hours and 30 minutes on a single CPU (i5 2500k). For users with GPUs with compute capability >= 3.0 (e.g., GTX 650 and newer), using TensorFlow-GPU with Keras will significantly speed up the computation process.
