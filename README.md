# Fashion-MNIST Classification with TensorFlow
This project is a lab exercise from **Introduction to Computer Vision with TensorFlow**.  
The goal of this lab is to build and experiment with a simple neural network for image classification using the **Fashion-MNIST dataset**.
---
## Project Overview
The project trains a neural network using **TensorFlow and Keras** to classify images of clothing items from the Fashion-MNIST dataset.
Fashion-MNIST contains:
- 70,000 grayscale images
- Image size: 28 × 28 pixels
- 10 different clothing categories
Example classes include:
- T-shirt / Top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot
The model learns to identify patterns in the images and predict which clothing category each image belongs to.
---
## Neural Network Architecture
The baseline model used in this lab is a simple feedforward neural network:
- **Flatten layer** converts 28×28 images into a 784-element vector.
- **Dense layer (64 neurons)** learns features from the input.
- **Softmax output layer (10 neurons)** predicts the probability for each clothing class.
---
## Experiments
Several experiments were conducted to understand how neural networks behave.
### Exercise 1
Changed the number of neurons in the hidden layer.
### Exercise 2
Added an additional hidden layer to test deeper networks.
### Exercise 3
Removed input normalization to observe the impact on training accuracy.
### Exercise 4
Removed the `Flatten()` layer to demonstrate shape mismatch errors.
### Exercise 5
Changed the number of neurons in the output layer to show why it must match the number of classes.

---
## Requirements
Install the required libraries: requirements.txt 
## Dataset
Fashion-MNIST dataset:  
https://github.com/zalandoresearch/fashion-mnist

