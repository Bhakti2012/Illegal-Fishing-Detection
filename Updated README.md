# Illegal-Fishing-Detection
This repository contains code for a machine learning model trained to classify fishing activity based on vessel data collected from the Global Fishing Watch. The model uses the various algorithm for classification and achieves an highest accuracy of 99.78% on the training data.

## Prerequisites
Make sure you have the following libraries installed:
* pandas
* scikit-learn
* xgboost
* joblib

## Data Collection
The data used for training and testing the model is obtained from the Global Fishing Watch website. The dataset is in CSV format and contains information about vessel activities such as speed over ground, distance from shore, distance from port, time elapsed, latitude, longitude, and source.

## Data Pre-processing
The AIS (Automatic Identification System) data is pre-processed to clean and transform it into a suitable format for training the model. The pre-processing steps include:
* Dropping rows with missing values
* Converting the timestamp to a datetime object
* Creating additional features like 'speed_over_ground', 'distance_from_shore', 'distance_from_port', 'time_elapsed', 'lat', 'lon', and 'activity'
* Labeling the 'activity' column based on the 'is_fishing' column
* Saving the pre-processed data to a CSV file named "Final_data.csv"

## Model Training
The pre-processed data is split into training and testing datasets using the train_test_split function from scikit-learn. The features ('speed_over_ground', 'distance_from_shore', 'distance_from_port', 'time_elapsed', 'lat', 'lon') are scaled using the StandardScaler to normalize the data. The Random Forest classifier,Xgboost,Knn,Navie Bayes from scikit-learn is then  trained on the scaled training data. The trained model is saved to a file named "randomforest_model(9978).sav".

## Model Evaluation
The trained model is evaluated on the testing dataset by making predictions on the scaled testing features. The accuracy of the model is calculated using the accuracy_score function from scikit-learn and printed to the console. The training accuracy is also calculated and displayed.

## Model Testing
To test the trained model on new data, a separate dataset named "trollers.csv" is pre-processed following the same steps as for the training data. The pre-processed data is saved to a file named "test_data.csv". The saved Random Forest model is loaded using the joblib.load function from the joblib library. A sample vessel data is created, and the loaded model is used to predict the activity based on the vessel data. The prediction result is printed to the console.
