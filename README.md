🐱🐶 Cat and Dog Image Classifier - README
📌 Project Overview
This project is a Convolutional Neural Network (CNN) model that classifies images of cats and dogs using TensorFlow 2.0 and Keras. The goal is to build a classifier that achieves at least 63% accuracy on a test dataset.



📂 Dataset Structure
Ensure your dataset is structured as follows:


cats_and_dogs/
├── train/
│   ├── cats/        # Training images of cats
│   │   ├── cat.0.jpg
│   │   ├── cat.1.jpg
│   ├── dogs/        # Training images of dogs
│   │   ├── dog.0.jpg
│   │   ├── dog.1.jpg
├── validation/
│   ├── cats/        # Validation images of cats
│   ├── dogs/        # Validation images of dogs
├── test/
│   ├── all/         # Unlabeled test images
│   │   ├── 1.jpg
│   │   ├── 2.jpg
