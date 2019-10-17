# Chat Toxicity Classifier
Classifier to get toxicity metrics

This is a Flask application that spits the probability of a comment entered, being of following categories: Toxic, Severe Toxic, Obscene, Threat, Insult, Identity Hate.

The model was created using keras and ipython notebook for the same is in the scripts folder. 

Following are the steps followed in the notebook:

1. The data was taken from kaggle Toxic Comment Classification Challenge (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) dataset.
2. Preprocessing is done on train and test data to turn comments into word index of equal length by truncation and padding.
3. Using word -> vector of Glove a embedding matrix is created.
4. A simple bidirectional LSTM with 2 fully connected layer is created.

The models or pickeled objects are not in models folder since it would increase the size of repository, but it can be easily created by running the notebook.

LIVE DEMO HERE -> https://chattoxicity.appspot.com/

# Screenshot

![image](https://user-images.githubusercontent.com/16362957/66312699-ab04c880-e900-11e9-9c31-ca3f671a0107.png)

![image](https://user-images.githubusercontent.com/16362957/66312796-d2f42c00-e900-11e9-8e1d-48dabb8c6062.png)

# Dependencies
Flask, Tensorflow, Keras

# References
https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout?sortBy=relevance&group=everyone&search=toxic+comment+&page=1&pageSize=20&turbolinks%5BrestorationIdentifier%5D=e88bae67-bc31-400d-a502-053b547cb912

# My Original Contribution & Learnings

Contribution =>
Reimplemented the code after understanding the above kaggle kernel mentioned in the reference.
Used the generated model in a flask application which was built so that prediction for an input can be made interactive.
Deployed on GCP using App Engine.

Major Learnings => 
Learnt how to use transfer learning using Glove.
Learnt how to built Flask application and serve a saved keras model.
Learnt how to deploy on GCP using App Engine.
