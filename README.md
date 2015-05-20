# Kaggle's Otto Competition

Code used for my submission to the Otto Kaggle competition, classifying products with 93 features into 9 product categories. These models achieved a logloss of 0.424 on public and private leaderboards. This was in the top 10% of submissions, though I didn't spend much time tuning and used a uniformly weighted ensemble as I was writing up my thesis!

The two models used were:
* XGBoost GBDT (xgboost.R)
* H2O DNN (h2o-dnn.R)

Train and test datasets are restricted to competition participants:  
https://www.kaggle.com/c/otto-group-product-classification-challenge/
