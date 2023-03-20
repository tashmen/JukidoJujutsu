# Jukido Jujutsu Stance Application
This is a hobby project that was built for fun to see what could be accomplished with pose recognition software.

Fundamentally the application contains two parts:
1. PoseClassifier - Python Project
2. Stances App - Basic HTML/Javascript web application

The web application is currently deployed on a free server located here: http://jukidojujutsu.42web.io/

##Pose Classifier
The pose classifier contains a number of python scripts that can be executed individually.  Each script performs a particular transformation that ultimately converts videos into a neural network that can predict stances.

split_video.py - Splits a video into individual images
dataaugmentation.py - Generates new images from the split images with zoom and mirror transformations
mediapipe_buildcsv.py - Converts the images into poses by [link](https://google.github.io/mediapipe/solutions/pose.html "using mediapipe's pose estimation model").  The data is stored into a csv file.
mediapipe_classifier.py - Uses the CSV files to train a neural network using keras and then converts the model to tflite format.

##Stances App
The stances app is a basic html application that uses mediapipe and tensorflow to predict stances using the tflite models that are constructed from the pose classifier.

The application runs in two different modes:
1. Tell - Predicts which stance a user is in based on camera input
2. Show - Allows the user to select a stance and then the user should try to match the stance shown and hold it for 5 seconds