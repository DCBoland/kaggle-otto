# Original code post: https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/12947/achieve-0-50776-on-the-leaderboard-in-a-minute-with-xgboost
# Updated with feature engineering, new parameters and model averaging.

require(xgboost)
require(methods)
library(caret)
library(dplyr)
library(tidyr)

train = read.csv('train.csv',header=TRUE,stringsAsFactors = F)
test = read.csv('test.csv',header=TRUE,stringsAsFactors = F)
train = train[,-1]
test = test[,-1]

train <- train[sample(nrow(train)),]

y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1

x = rbind(train[,-ncol(train)],test)
colnames(x) <- as.numeric(gsub('feat_','',colnames(x)))

# Anscombe transform
x <- sqrt(x+3/8)

# Prepare for XGBoost
x = as.matrix(x)
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 16,
              "eta" = 0.01,
              "max_depth" = 10,
              "min_child_weight" = 4,
              "subsample" = .5,
              "colsample_bytree" = .5)

#####

submission <- read.csv("sampleSubmission.csv")
submission[,2:10] <- 0

nround = 3475

# Average over 10 models
for(i in 1:10){
  print(i)
  # Train the model 
  bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround, verbose=F)
  
  # Make prediction
  pred = predict(bst,x[teind,])
  pred = matrix(pred,9,length(pred)/9)
  pred = t(pred)
  
  # Output submission
  pred = data.frame(1:nrow(pred),pred)
  names(pred) = c('id', paste0('Class_',1:9))
  
  submission[,2:10] <- submission[,2:10] + pred[,2:10]
  print(i)
  write.csv(submission,file="submission.csv",row.names=FALSE) 
}