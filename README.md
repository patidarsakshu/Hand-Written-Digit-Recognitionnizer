# Hand-Written-Digit-Recognition
This repository contains code for handwriting recognition using Convolutional Neural Networks (CNN). The implementation is based on deep learning models to classify writers based on their individual writing styles. The IAM Handwriting Dataset is utilized for training and testing the models.

## Overview
This project implements a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset. The model is built with Keras and TensorFlow, achieving an accuracy of 99.671% on the test data. This repository includes code for data preparation, model training, and result prediction.

## Table of Contents
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [References](#references)

## Project Structure
- `data/`: Contains the dataset files (`train.csv` and `test.csv`).
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model evaluation.
- `src/`: Contains Python scripts for data preprocessing, model definition, and training.
- `results/`: Contains the output predictions and submission files.
- `README.md`: This file.

## Data Preparation
The MNIST dataset is loaded from CSV files and preprocessed as follows:
1. **Loading Data:**
   ```python
   import pandas as pd
   
   train = pd.read_csv("data/train.csv")
   test = pd.read_csv("data/test.csv")

 2.  Normalization and Reshaping:


    X_train = train.drop(labels=["label"], axis=1) / 255.0
    Y_train = pd.get_dummies(train["label"])
    X_train = X_train.values.reshape(-1, 28, 28, 1)
    test = test / 255.0
    test = test.values.reshape(-1, 28, 28, 1)

3. Model Architecture

The CNN model is defined with the following layers:

   3.1. Convolutional Layers:
        Conv2D layers for feature extraction.
        MaxPool2D for down-sampling.
        Dropout for regularization.
   3.2.  Fully Connected Layers:
        Dense layers for classification.


    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

    model = Sequential([
    Conv2D(32, (5,5), activation='relu', input_shape=(28,28,1), padding='same'),
    Conv2D(32, (5,5), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2,2)),
    Dropout(0.25),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
    ])


4. Training the Model

The model is trained with data augmentation to prevent overfitting:

  5. Data Augmentation:

  from keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
    )
    datagen.fit(X_train)

6. Model Training:

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(datagen.flow(X_train, Y_train, batch_size=86),
    epochs=1, validation_split=0.1, verbose=2)

7. Results

The model achieves an accuracy of 99.671% on the MNIST test dataset. The results are saved in a CSV file for submission.
**How to Run**

**Install Dependencies**

bash

pip install -r requirements.txt

Run the Jupyter Notebook:

bash

    jupyter notebook notebooks/model_evaluation.ipynb

Dependencies

    keras
    tensorflow
    pandas
    numpy
    matplotlib
    scikit-learn

References

    MNIST Dataset
    Keras Documentation
    TensorFlow Documentation

