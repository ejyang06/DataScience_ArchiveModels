###########################################################
# Sample code to Predict Titanic
# Questions & concerns email:  Taposh Dutta Roy
# Example : Predictive Model for Titanic
###########################################################
# Sample
# This is an example to develop a predictive model using R
# ----------------------------------------------------------

##load packages
packages <- c("randomForest", "rpart")
packages <- lapply(packages, FUN = function(x) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x)
    suppressPackageStartupMessages(library(x, character.only = TRUE, warn.conflicts = FALSE, quietly = TRUE))
  }
})

#libraries
library(randomForest)
library(rpart)

set.seed(415)

#load data
train <- read.csv("/tmp/train.csv")
test <- read.csv("/tmp/test.csv")

#Feature Engineering
feature_eng <- function(train_df, test_df) {
  # Combining the train and test sets for purpose engineering
  test_df$Survived <- NA
  combi <- rbind(train_df, test_df) 
  
  #Features engineering
  combi$Name <- as.character(combi$Name)
  
  # The number of titles are reduced to reduce the noise in the data
  combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
  combi$Title <- sub(' ', '', combi$Title)
  #table(combi$Title)
  combi$Title[combi$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
  combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
  combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
  combi$Title <- factor(combi$Title)
  
  # Reuniting the families together
  combi$FamilySize <- combi$SibSp + combi$Parch + 1
  combi$Surname <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
  combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep="")
  combi$FamilyID[combi$FamilySize <= 2] <- 'Small'
  #table(combi$FamilyID)
  combi$FamilyID <- factor(combi$FamilyID)
  
  
  # Decision trees model to fill in the missing Age values
  Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize, data=combi[!is.na(combi$Age),], method="anova")
  combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])
  
  # Fill in the Embarked and Fare missing values
  #which(combi$Embarked == '')
  combi$Embarked[c(62,830)] = "S"
  combi$Embarked <- factor(combi$Embarked)
  #which(is.na(combi$Fare))
  combi$Fare[1044] <- median(combi$Fare, na.rm=TRUE)
  
  # Creating a new familyID2 variable that reduces the factor level of falilyID so that the random forest model
  # can be used
  combi$FamilyID2 <- combi$FamilyID
  combi$FamilyID2 <- as.character(combi$FamilyID2)
  combi$FamilyID2[combi$FamilySize <= 3] <- 'Small'
  combi$FamilyID2 <- factor(combi$FamilyID2)
  
  return(combi)
}

# Splitting back to the train and test sets
data <- feature_eng(train, test)
train <- data[1:891,]
test <- data[892:1309,]

# Fitting random forest to the train set, predicting in the test set and creating the submitting file
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize +
                      FamilyID2, data=train, importance=TRUE, ntree=2000)

#Variable importance plot
varImpPlot(fit)

#Prediction
Prediction <- predict(fit, test)

#Submission file
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)

#Write output file 
write.csv(submit, file = "submission.csv", row.names = FALSE)