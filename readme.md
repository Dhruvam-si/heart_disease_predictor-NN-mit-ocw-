mit_assignments

My work through the deep learning assignments from MIT OpenCourseWare, built in Keras. Each project is a hands-on exercise in reading training results (accuracy, loss, the train/validation gap) and deciding what to change next.

Inspired by, and grateful to, Prof. Rama Ramakrishnan's MIT OCW course.

Projects
Project	Task	Best result
Heart Disease Predictor	Binary classification from 29 biomarkers	88% test accuracy
Fashion-MNIST CNN	10-class image classification	91.75% test accuracy
1. Heart Disease Predictor

A small neural network that predicts the presence of heart disease from patient biomarkers.

Data: 300+ patients, 29 biomarker features (dataset provided with the assignment)
Model: input of 29 features, one hidden layer of 16 neurons (ReLU), one sigmoid output for the yes/no prediction
Training: binary cross-entropy loss, Adam optimizer, random seed 42 for reproducibility
Results: test accuracy 88% (loss 0.28); best training accuracy 91% (loss 0.24)

What it taught me: how accuracy, loss, and the number of training epochs relate, even on a very simple model.

2. Fashion-MNIST CNN

A convolutional network that classifies 28×28 grayscale clothing images into 10 categories (60,000 training / 10,000 test images).

Best model: two convolutional blocks (32 and 64 filters) with max pooling, followed by dense layers of 64, 128, and 128 neurons with dropout (0.3) and a softmax output
Result: 91.75% test accuracy, with train and test accuracy within about a point of each other

Experiments I ran (one change at a time, predicting the outcome before each run):

Reading a confusion matrix: most errors were between look-alike upper-body items (Shirt, T-shirt, Pullover, Coat)
Diagnosing overfitting vs. underfitting from training and validation curves
Replacing Flatten + Dense with global average pooling, which removed overfitting but also cut accuracy
Data augmentation: strong settings (vertical flips, large rotations) hurt; mild ones were neutral
Dropout, learning-rate decay with ReduceLROnPlateau, and padding="same" to keep deeper stacks from shrinking the image to nothing
Comparing against a dense-only network (88%) to see what convolutions add
Tech stack
Python
Keras (model building and training)
NumPy (array handling)
pandas (organizing predictions, confusion matrices)
Matplotlib (training curves)
Running a project
bash
pip install keras numpy pandas matplotlib
python mit_cnn.py
Related
ML_network-in-C-from-Scratch: a separate repo with a neural network written from scratch in C++
What I'm learning from this

Improving a model has no single fixed recipe. The loop that worked: look at the train/validation numbers, form a hypothesis about what's limiting the model, change one thing, and compare against a fixed baseline.
