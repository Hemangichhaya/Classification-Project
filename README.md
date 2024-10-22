# Vegetable Image Classification - Ninjacart
## Problem Statement
Ninjacart is India's largest fresh produce supply chain company, solving complex supply chain challenges by leveraging innovative technology. They source fresh produce from farmers and deliver it to businesses within 12 hours. A key part of their automation process is developing robust classifiers to distinguish between images of various vegetables and identify non-vegetable images as noise.

We are tasked with developing a multiclass classifier to identify images of onions, potatoes, tomatoes, and market scenes using a dataset provided by Ninjacart.

## Dataset
The dataset contains images scraped from the web and is divided into train and test folders, each containing sub-folders for onions, potatoes, tomatoes, and market scenes (noise).

Dataset Link: Ninjacart Vegetable Dataset

## Train Folder
Tomato: 789 images
Potato: 898 images
Onion: 849 images
Indian market (noise): 599 images
---
## Test Folder
Tomato: 106 images
Potato: 83 images
Onion: 81 images
Indian market (noise): 81 images
Objective
Develop a program that can recognize vegetable items (tomato, potato, onion) in photos and classify them correctly, while identifying non-vegetable market images as noise.
---
## Concepts Tested
Dataset Preparation & Visualization
CNN Models
Implementing Callbacks
Handling Overfitting
Transfer Learning
---
## Process
1. Libraries
We used the following key libraries:

TensorFlow & Keras for deep learning
Matplotlib & Seaborn for data visualization
OpenCV for image processing
2. Data Download & Visualization
Download and unzip the dataset using the provided link.
Visualize the data and plot a sample of images for each class using TensorFlow or Matplotlib.
Verify the image count and check the dimensions to ensure consistency.
3. Data Preprocessing
Perform an 80-20 train-validation split for hyperparameter tuning.
Resize all images to uniform dimensions and rescale pixel values to [0-1].
Apply data augmentation techniques to handle class imbalance.
4. Model Selection & Training
Multiple models (VGG16, ResNet50, MobileNet) were used for training.
Transfer learning with pre-trained weights was applied due to the small size of the dataset.
Batch normalization and dropout layers were added to address overfitting.
5. Callbacks
Implemented TensorBoard to track training metrics.
Used EarlyStopping and ModelCheckpoint callbacks to optimize the training process.
6. Evaluation
Compared model performance using metrics such as Accuracy, Precision, Recall, and F1 Score.
Visualized results using confusion matrices and plotted training accuracy and loss graphs.
Evaluation Criteria
Task	Points
Dataset Import & Exploration	10 points
Exploratory Data Analysis	20 points
Class Distribution & Image Dimensions Plotting	10 points
Train-Validation-Test Split	10 points
Model Architecture & Training	50 points
Baseline CNN Classifier	10 points
Reducing Overfitting	10 points
Callbacks Implementation	10 points
Transfer Learning	10 points
Model Metrics & Confusion Matrix	10 points
Testing on Test Set	20 points
Summary & Insights	10 points
Key Techniques Used
1. Transfer Learning
We fine-tuned pre-trained models like VGG16, ResNet, and MobileNet to improve classification accuracy.

2. Data Augmentation
To address the class imbalance, we applied data augmentation techniques such as rotation, zooming, and horizontal flips.

3. Handling Overfitting
Implemented BatchNormalization, Dropout layers, and EarlyStopping callbacks to prevent overfitting.

Results
The best model achieved an accuracy of XX% on the test set.
Confusion matrices were used to evaluate the model's performance across different classes.
Insights
Transfer learning significantly improved model performance due to the small size of the dataset.
Data augmentation and regularization techniques were essential in reducing overfitting.
Conclusion
This project successfully developed a robust multiclass classifier to identify vegetables and noise in the dataset. The use of transfer learning and data augmentation techniques improved model performance, and future improvements could focus on expanding the dataset or applying attention mechanisms.

