Project Name: Deep Neural Network for Classification of Patient Data

Research question: Can a Deep Neural Network be constructed using Patient
Data to predict if a patient will have diabetes?

My Hypothesis: A Deep Neural Network Model can be constructed from the Patient
data.

Data collected from 
https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

Results:
A Deep Neural Network was able to be created using three layers of Dense 100
that gave an Accuracy of 96% with a Precision of 76% and a Recall of 75%.
Better Precision was able to be created with a Dense 100, Dense 90, Dense 90, model
but Recall was lower at Accuracy: 97%, Precision 99% and Recall of 67%.

Because Recall is penalized whenever a false negative is predicted, our 100/90/90 model
had more false negatives than our 100/100/100 model, but also less false positives.
Best weights for each model are contained in the folder, and the code for the model would
need to be updated depending on which model was preferred.
