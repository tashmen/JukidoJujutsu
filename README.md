# Jukido Jujutsu Stance Application
This is a hobby project that was built to have something unique to discuss and show off as part of my black belt testing.  I was also curious to see what could be accomplished with pose recognition software.  A lot of code was borrowed and adapted from other example projects:
* https://medium.com/@leahnagy/yoga-pose-classification-with-tensorflows-movenet-model-3e5771fda292
* https://www.kaggle.com/code/venkatkumar001/yoga-pose-recognition-mediapipe


Fundamentally the application contains two parts:
1. **PoseClassifier** - Python Project
2. **Stances App** - Basic HTML/Javascript web application

The web application is currently deployed on a free server located here: https://jukidojujutsu.42web.io/

## Pose Classifier
The pose classifier contains a number of python scripts that can be executed individually.  Each script performs a particular transformation that ultimately converts videos into a neural network that can predict stances.

* **split_video.py** - Splits a video into individual images
* **dataaugmentation.py** - Generates new images from the split images with zoom and mirror transformations (experimental)
* **hyperparametertuning.py** - Finds the optimal model to use for best results (WiP)
* **mediapipe_buildcsv.py** - Converts the images into poses [using mediapipe's pose estimation model](https://google.github.io/mediapipe/solutions/pose.html).  The data is stored as a CSV file.
* **mediapipe_classifier.py** - Uses the CSV files to train a neural network using keras and then converts the model to tflite format.

## Stances App
The stances app is a basic html application that uses mediapipe and tensorflow to predict stances using the tflite models that are constructed from the pose classifier.

The application runs in two different modes:
1. **Tell** - Predicts which stance a user is in based on camera input
2. **Show** - Allows the user to select a stance and then the user should try to match the stance shown and hold it for 5 seconds


## Additional Notes
All images used to train the models are not included for privacy reasons. (Special thanks to all the dojo members who took some time to hold the stances so that the models could be trained with real people.)

* The app is on a free server which has a 50,000 hit limit per day
  * I assume that it will display some error message if we hit the limit and people would need to wait until the following day
* For best results
  * Use Chrome - other browsers like Safari may not be able to read the camera 
  * Stand more or less facing the camera in the same orientation as the images in the app
    * The app can to some degree tell stances at slight angles, but the accuracy is not as good
    * The app will not recognize opposite facing stances (e.g. left vs right zenkutsu)
  * Stand at a similar distance from the camera as the images in the app
  * Small children should stand closer to the camera although they may have difficulty using the app regardless
    * Likewise, if someone is excessively tall then standing further from the camera may improve accuracy
* Explanation of the output
  * Tell
    * Unknown Stance -> This will display if the app cannot positively identify the stance
      * Best Guess is what stance the app believes you are currently closest to
    * Ambiguous Stance -> This will display if the app believes you are standing between two stances and it will list the stances that it believes you could potentially be standing in
      * Predicted is what stance the app believes you are currently closest to which may or may not be one that was positively identified
    * Matched Stance -> This will display if the app positively identifies only a single stance and it is the closest one
    * Percentages
      * Parenthesis "()" is the match rating for the stance that is the closest 
      * Brackets "[]" is the match rating for a positive identification where 100% is an exact match
  * Show
    * Red outline -> Stance does not match the one shown
    * Yellow outline -> Stance matches and must be held for 5 seconds
      * Also displayed if the stance matches by closeness at less than 80%
    * Green outline -> Stance matches, has been held for longer than 5 seconds and matches higher than 80%