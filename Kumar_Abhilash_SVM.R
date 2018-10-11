
# Install the required packages
requiredPackages = c('gridExtra','readr','caret','kernlab','ggplot2','dplyr')

#Installing the required packages
for(item in requiredPackages){
  if(!require(item,character.only = TRUE)){
    install.packages(item)
  }
  library(item, character.only = TRUE)
}

# Loading the train and test data
mnist_train <- read.csv("mnist_train.csv", stringsAsFactors = F, header = FALSE)
mnist_test <- read.csv("mnist_test.csv", stringsAsFactors = F, header = FALSE)

# DAta Understanding
# In our data set, we have 28*28 pixel, that's why 784 attributes + 1 the digit label.
# We are supposed to develop a model using Support Vector Machine which should correctly 
# classify the handwritten digits based on the pixel values given as features.

# Let's sample the test data and take only 15% i.e. 9000 records for our analysis
mnist_train <- mnist_train[sample(nrow(mnist_train), 9000), ]
str(mnist_train)      # 9000 obs. of  785 variables
str(mnist_test)       # 10000 obs. of  785 variables

dim(mnist_train)
dim(mnist_test)

# Let's rename the first column as Label
colnames(mnist_train)[1] <- "Label"
colnames(mnist_test)[1] <- "Label"

# Let's check for NA values
as.numeric(sapply(mnist_train, function(x) sum(is.na(x))))
as.numeric(sapply(mnist_test, function(x) sum(is.na(x))))
# We don't have any NA values in the test and train data

#Making our target labels to factor and assign the exiting objects to train and test 
train <- mnist_train
test <- mnist_test
train$Label <- factor(train$Label)
test$Label <- factor(test$Label)
str(train)
str(test)

set.seed(100)

# Constructing Model

# 1) Linear model - SVM  at Cost(C) = 1
#####################################################################
# Note: While executing the ksvm we will get warning messages. We can ignore that.
# Model with C =1
model_1 <- ksvm(Label ~ ., data = train, scale = FALSE, C=1)

# Predicting the model results on test data
evaluate_1 <- predict(model_1, test)

# Confusion Matrix - Finding accuracy, Sensitivity and specificity
confusionMatrix(evaluate_1, test$Label)

# Accuracy    : 0.9592
#                       Class:0  Class:1 Class:2  Class:3  Class:4  Class:5  Class:6  Class:7  Class:8  Class:9
#Sensitivity            0.9878   0.9894   0.9554   0.9525   0.9603   0.9451   0.9676   0.9484   0.9497   0.9316
#Specificity            0.9962   0.9971   0.9943   0.9945   0.9948   0.9960   0.9967   0.9959   0.9941   0.9950
#--------------------------------------------------------------------

# Linear model - SVM  at Cost(C) = 10
#####################################################################

# Model with C =10.
model_10 <- ksvm(Label ~ ., data = train, scale = FALSE, C=10)

# Predicting the model results on test data
evaluate_10 <- predict(model_10, test)

# Confusion Matrix - finding accuracy, sensitivity and specificity
confusionMatrix(evaluate_10, test$Label)

# Accuracy    : 0.9683
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
#Sensitivity            0.9888   0.9894   0.9719   0.9653   0.9644   0.9596   0.9729   0.9572   0.9600   0.9504
#Specificity            0.9970   0.9980   0.9949   0.9960   0.9959   0.9967   0.9976   0.9971   0.9956   0.9961

#Using Linear Kernel
#####################################################################
Model_linear <- ksvm(Label~ ., data = train, scale = FALSE, kernel = "vanilladot")
Model_linear
# cost C = 1
# Number of Support Vectors : 2564 

# Train accuracy 
Eval_linear_train <- predict(Model_linear, train)
#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear_train, train$Label)      # Accuracy : 1

# Test accuracy 
Eval_linear_test <- predict(Model_linear, test)
#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear_test, test$Label)       # Accuracy : 0.9176 


#Using RBF Kernel
#####################################################################
Model_RBF <- ksvm(Label~ ., data = train, scale = FALSE, kernel = "rbfdot")
Model_RBF
# cost C = 1 
# sigma =  1.63772199359522e-07 
# Number of Support Vectors : 3552 

# Train accuracy:
Eval_RBF_train <- predict(Model_RBF, train)
#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF_train,train$Label)        # Accuracy : 0.9801

# Test accuracy:
Eval_RBF_test <- predict(Model_RBF, test)
#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF_test,test$Label)         # Accuracy : 0.9592


############   Hyperparameter tuning and Cross Validation #####################

# We will use the train function from caret package to perform Cross Validation. 

# traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 3 implies Number of folds in CV.

trainControl <- trainControl(method = "cv", number = 3, verboseIter=TRUE)

# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

# Expand.grid functions takes set of hyperparameters, that we shall pass to our model.
# Tuning the sigma value sigma =  1.63772199359522e-07 +- 1 and using hit and trail method for C value.
# We are going to consider the below values for sigma nad C and then test the optimized value for building and testing our model
# Sigma : 0.63e-7, 1.63e-7, 2.63e-7
# C : 1, 2, 3
grid <- expand.grid(.sigma = c(0.63e-7,1.63e-7,2.63e-7),.C=c(1,2,3))

# train function takes Target ~ Prediction, Data, Method = Algorithm
# Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit_RBF.svm <- train(Label~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

fit_RBF.svm

print(fit_RBF.svm)
# Support Vector Machines with Radial Basis Function Kernel 
# 
# 9000 samples
# 784 predictor
# 10 classes: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 
# 
# No pre-processing
# Resampling: Cross-Validated (3 fold) 
# Summary of sample sizes: 6001, 6000, 5999 
# Resampling results across tuning parameters:
#   
#   sigma     C  Accuracy   Kappa    
# 6.30e-08  1  0.9304442  0.9226680
# 6.30e-08  2  0.9386663  0.9318101
# 6.30e-08  3  0.9395554  0.9327981
# 1.63e-07  1  0.9509998  0.9455245
# 1.63e-07  2  0.9575555  0.9528122
# 1.63e-07  3  0.9593333  0.9547881
# 2.63e-07  1  0.9601110  0.9556542
# 2.63e-07  2  0.9626667  0.9584951
# 2.63e-07  3  0.9635555  0.9594836
# 
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were sigma = 2.63e-07 and C = 3.
plot(fit_RBF.svm)
# The optimized values for hyperparameters are sigma = 2.63e-07 and C = 3

############ Buidling RBF model with optimised sigma and c value ############
# Lets build a model with C = 3 and sigma = 2.63e-07
RBF_model_final <- ksvm(Label~.,
                       data=train,
                       kernel="rbfdot",
                       scale=FALSE,
                       C=3,
                       kpar=list(sigma=2.63e-7))
RBF_model_final
# parameter : cost C = 3 
# Hyperparameter : sigma =  2.63e-07 
# Number of Support Vectors : 3727

# Train accuracy 
Eval_RBF_train_final <- predict(RBF_model_final, train)
confusionMatrix(Eval_RBF_train_final,train$Label)
# Net Train Accuracy = 0.9989

# Test accuracy 
Eval_RBF_test_final <- predict(RBF_model_final, test)
confusionMatrix(Eval_RBF_test_final,test$Label)
# Net Train Accuracy = 0.9707

# Conclusion:
# The model demonstrates a test accuracy of .97 when sigma = 2.63e-07 and C = 3
