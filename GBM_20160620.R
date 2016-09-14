########################################
# Author: Taposh Roy
# High Utilizer h2o Ensemble Code in R      
########################################

library(h2o)
attach("/hu/input/hu.RData")
localH2O = h2o.init(ip = '172.16.14.233', port = 54321,strict_version_check= FALSE)
#######################
#Submitter function 
######################
submitter <- function(id,predictions,filename)
{ 
  submission<-cbind(id,predictions)
  colnames(submission) <- c("mbr_id", "prediction")
  submission <- as.data.frame(submission)
  filename = paste0("/hu/output/",filename,"l103295_ensemble.csv")
  write.csv(submission, filename,row.names = FALSE)
}

#####################################################
# Splitting into Training set and Validation set for model building
#####################################################
trainrows =nrow(hutrain)/3 # 2/3 train and 1/3 validation
train = subset(hutrain,mbr_id>=trainrows )
val = subset(hutrain,mbr_id<trainrows)


#####################################################
#Initialize h2o
#####################################################
library(h2oEnsemble)  # This will load the `h2o` R package as well
library(cvAUC) # cross validation AUC
h2o.removeAll() # Clean slate - just in case the cluster was already running
localH2O = h2o.init(ip = '172.16.14.239', port = 54323,strict_version_check= FALSE,nthreads = -1)

# Load into h2o
train_hex = as.h2o(train)
val_hex = as.h2o(val)
test_hex = as.h2o(hutest)

# Define dependent and independent variables
y <- "hu_01"
x <- train_hex[,-c(141:145)]
x1 <- setdiff(names(x), y)

## For binary classification, the response should be encoded as factor (also known as the enum type in Java).
train_hex[,y] <- as.factor(train_hex[,y])  

#####################################################################
# Random Forest - I do this to get the important variables
#####################################################################
hu.rf = h2o.randomForest(y = y, x = x1, training_frame = train_hex)

#Lets look at variables since we will get penalized if used all variables
impvariables = h2o.varimp(hu.rf)

#Select the 12 variables important for ensembles
x2 = c("pri_cst","ethnct_ds_tx","dx_prspct_med_risk_qt","rx_prspct_ttl_risk_qt","mcare_prspct_med_risk_qt","loh_prspct_qt","cms_hcc_130_ct",
       "rx_inpat_prspct_ttl_risk_qt","cg_2014","cops2_qt","rx_prspct_ttl_risk_qt_p","mcare_prspct_med_risk_qt_p")

x2 = c("pri_cst","ethnct_ds_tx","dx_prspct_med_risk_qt","rx_prspct_ttl_risk_qt","mcare_prspct_med_risk_qt","loh_prspct_qt","cms_hcc_130_ct",
       "rx_inpat_prspct_ttl_risk_qt","cg_2014","cops2_qt","rx_prspct_ttl_risk_qt_p","mcare_prspct_med_risk_qt_p",
       "dx_cncur_med_risk_qt","age_yr_nb","ndi","race_ds_tx","loh_prspct_qt_p","rx_cncur_med_risk_qt",
       "hosp_day_ct_pri",   "rx_elig_mm_ct","cops_qt_p","dx_prspct_med_risk_qt_p","rx_inpat_prspct_ttl_risk_qt_p",
       "rx_cncur_med_risk_qt_p")

#########################################################################
# ENSEMBLE - Stacking
# http://learn.h2o.ai/content/tutorials/ensembles-stacking/
# Research - http://www.stat.berkeley.edu/~ledell/research.html
#########################################################################

# Define Baselearners and meta-leaners
learner <- c("h2o.glm.wrapper", "h2o.randomForest.wrapper", 
             "h2o.gbm.wrapper")
metalearner <- "h2o.glm.wrapper"


# Train Ensembles
fit <- h2o.ensemble(x = x2, y =y, 
                    training_frame = train_hex, 
                    family = "binomial", 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 5))

# Predict
pred <- predict(fit, val_hex)
#third column is P(Y==1)
predictions <- as.data.frame(pred$pred)[,3]
labels <- as.data.frame(val_hex[,y])[,1]

###############
# Basic Ensemble Evaluation
##############
cvAUC::AUC(predictions = predictions, labels = labels)

###############
# Check how each learner did 
##############
L <- length(learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(pred$basepred)[,l], labels = labels)) 
data.frame(learner, auc)

#          learner       auc
#1          h2o.glm.wrapper 0.8779844
#2 h2o.randomForest.wrapper 0.8439109
#3          h2o.gbm.wrapper 0.8987327

##################################################################################################
# Now start specifying custom learners - You should tune here
##################################################################################################

#GLM with alpha =0 ridge penalty
h2o.glm.1 <- function(..., alpha = 0.0) 
{
  h2o.glm.wrapper(..., alpha = alpha)
}

#GLM with alpha =1 Lasso penalty
h2o.glm.2 <- function(..., alpha = 1.0) 
{
  h2o.glm.wrapper(..., alpha = alpha)
}

#Random Forest with 50 trees
h2o.randomForest.1 <- function(..., ntrees = 50, nbins = 50, seed = 786) 
{
  h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
}

#Random Forest with 100 trees and sample rate of 0.75
h2o.randomForest.2 <- function(..., ntrees = 100, sample_rate = 0.75, seed = 786)
{
  h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
}

#Random Forest with 100 trees and sample rate of 0.85 (varying the sample rate to see the impact)
h2o.randomForest.3 <- function(..., ntrees = 100, sample_rate = 0.85, seed = 786) 
{ 
  h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
}

#GBM specifying 100 trees and all default h2o parameters
h2o.gbm.1 <- function(..., ntrees = 100, seed = 786) 
{
  h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
}

#GBM specifying 100 trees, binning done (50 bins) and all default h2o parameters
h2o.gbm.2 <- function(..., ntrees = 100, nbins = 50, seed = 786) 
{
  h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
}

#GBM specifying 50 trees, max depth defined to be 10 and all default h2o parameters
h2o.gbm.3 <- function(..., ntrees = 50, max_depth = 10, seed = 786) 
{
  h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
}

#GBM specifying 50 trees, sample rate of 0.8, max depth =5 and all default h2o parameters
h2o.gbm.4 <- function(..., ntrees = 50, col_sample_rate = 0.8, max_depth = 10, seed = 786) 
{ 
  h2o.gbm.wrapper(..., ntrees = ntrees,  max_depth = max_depth,col_sample_rate = col_sample_rate, seed = seed)
}
#######################################################################################################
# Customized learner library
################################
learner <- c("h2o.glm.wrapper"
            ,"h2o.randomForest.1"
            ,"h2o.randomForest.2"
            ,"h2o.randomForest.3"
            ,"h2o.gbm.1"
            ,"h2o.gbm.2"
            ,"h2o.gbm.3"
            ,"h2o.gbm.4")

customfit <- h2o.ensemble(x = x2, y = y,
                          training_frame = train_hex,
                          learner = learner,
                          metalearner = metalearner,
                          cvControl = list(V = 5))
#######################################################################################################
#My learner
h2o.glm.1 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.2 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.3 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.randomForest.1 <- function(..., ntrees = 200, nbins = 50, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.randomForest.2 <- function(..., ntrees = 200, sample_rate = 0.75, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.3 <- function(..., ntrees = 200, sample_rate = 0.85, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.4 <- function(..., ntrees = 200, nbins = 50, balance_classes = TRUE, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, balance_classes = balance_classes, seed = seed)
h2o.gbm.1 <- function(..., ntrees = 100, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.gbm.2 <- function(..., ntrees = 100, nbins = 50, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.gbm.3 <- function(..., ntrees = 100, max_depth = 10, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.4 <- function(..., ntrees = 100, col_sample_rate = 0.8, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.5 <- function(..., ntrees = 100, col_sample_rate = 0.7, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.6 <- function(..., ntrees = 100, col_sample_rate = 0.6, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.7 <- function(..., ntrees = 100, balance_classes = TRUE, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, seed = seed)
h2o.gbm.8 <- function(..., ntrees = 100, max_depth = 3, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.deeplearning.1 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.2 <- function(..., hidden = c(200,200,200), activation = "Tanh", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.3 <- function(..., hidden = c(500,500), activation = "RectifierWithDropout", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.4 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, balance_classes = TRUE, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, balance_classes = balance_classes, seed = seed)
h2o.deeplearning.5 <- function(..., hidden = c(100,100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.6 <- function(..., hidden = c(50,50), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.7 <- function(..., hidden = c(100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)

###################################
# Customized learner library
################################
learner <- c("h2o.glm.wrapper"
             ,"h2o.randomForest.wrapper"
             ,"h2o.gbm.wrapper" 
             ,"h2o.glm.1"
             ,"h2o.glm.2"
             ,"h2o.glm.3"
             ,"h2o.randomForest.1"
             ,"h2o.randomForest.2"
             ,"h2o.randomForest.3"
             ,"h2o.randomForest.4"
             ,"h2o.gbm.1"
             ,"h2o.gbm.2"
             ,"h2o.gbm.3"
             ,"h2o.gbm.4"
             ,"h2o.gbm.5"
             ,"h2o.gbm.6"
             ,"h2o.gbm.7"
             ,"h2o.gbm.8"
             ,"h2o.deeplearning.1"
             ,"h2o.deeplearning.2"
             ,"h2o.deeplearning.3"
             ,"h2o.deeplearning.4"
             ,"h2o.deeplearning.5"
             ,"h2o.deeplearning.6"
             ,"h2o.deeplearning.7"
             )

customfit <- h2o.ensemble(x = x2, y = y,
                          training_frame = train_hex,
                          family='binomial',
                          learner = learner,
                          metalearner = metalearner,
                          cvControl = list(V = 5))

customfit_backup<-customfit
#######################
#Check on Validation datasets
valpred <- predict(customfit, val_hex)
#third column is P(Y==1)
valpredictions <- as.data.frame(valpred$pred)[,3]
labels <- as.data.frame(val_hex[,y])[,1]

#AUC expected
cvAUC::AUC(predictions = valpredictions, labels = labels) #0.8917869
###############
###########################################################################
# Check how each learner did so you can further tune 
##########################################################################

L <- length(learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(valpred$basepred)[,l], labels = labels)) 
data.frame(learner, auc)
############################################
#learner       auc
#1           h2o.glm.wrapper 0.8880615
#2  h2o.randomForest.wrapper 0.8388615
#3           h2o.gbm.wrapper 0.8969431
#4                 h2o.glm.1 0.8882188
#5                 h2o.glm.2 0.8880615
#6                 h2o.glm.3 0.8878074
#7        h2o.randomForest.1 0.8373203
#8        h2o.randomForest.2 0.8356392
#9        h2o.randomForest.3 0.8377248
#10       h2o.randomForest.4 0.8671202
#11                h2o.gbm.1 0.8975235
#12                h2o.gbm.2 0.9026275
#13                h2o.gbm.3 0.9005851
#14                h2o.gbm.4 0.9004320
#15                h2o.gbm.5 0.9002518
#16                h2o.gbm.6 0.9035129
#17                h2o.gbm.7 0.9000347
#18                h2o.gbm.8 0.9016446
#19       h2o.deeplearning.1 0.8133007
#20       h2o.deeplearning.2 0.8741026
#21       h2o.deeplearning.3 0.8866465
#22       h2o.deeplearning.4 0.8172911
#23       h2o.deeplearning.5 0.8873940
#24       h2o.deeplearning.6 0.8536418
#25       h2o.deeplearning.7 0.8741869
########################################
# Generate predictions on the test set:
#######################################
pred <- predict(customfit, test_hex)
predictions <- as.data.frame(pred$pred)[,3]

#Creating a submission frame
submitter(hutest$mbr_id,predictions,"husubmission-ensemble-baselined-4-")
########################################################################################################
#6/20/2016: using GBM for the ensemble

################################
localH2O = h2o.init(ip = '172.16.14.233', port = 54321,strict_version_check= FALSE)
library(h2oEnsemble)
train_hex = as.h2o(train)
val_hex = as.h2o(val)
test_hex = as.h2o(hutest)

# Define dependent and independent variables
y <- "hu_01"
x <- train_hex[,-c(141:145)]
x1 <- setdiff(names(x), y)

## For binary classification, the response should be encoded as factor (also known as the enum type in Java).
train_hex[,y] <- as.factor(train_hex[,y])  

x12 = c("pri_cst","ethnct_ds_tx","dx_prspct_med_risk_qt","rx_prspct_ttl_risk_qt","mcare_prspct_med_risk_qt","loh_prspct_qt","cms_hcc_130_ct",
       "rx_inpat_prspct_ttl_risk_qt","cg_2014","cops2_qt","rx_prspct_ttl_risk_qt_p","mcare_prspct_med_risk_qt_p")
GBM_learner <- c(
             "h2o.gbm.wrapper" 
             ,"h2o.gbm.1"
             ,"h2o.gbm.2"
             ,"h2o.gbm.3"
             ,"h2o.gbm.4"
             ,"h2o.gbm.5"
             ,"h2o.gbm.6"
             ,"h2o.gbm.7"
             ,"h2o.gbm.8"
            
)
GBM_metalearner <- "h2o.gbm.wrapper"

#####
GBM_customfit <- h2o.ensemble(x = x12, y = y,
                          training_frame = train_hex,
                          family='binomial',
                          learner = GBM_learner,
                          metalearner = GBM_metalearner,
                          cvControl = list(V = 5))

#######################
#Check on Validation datasets
GBM_valpred <- predict(GBM_customfit, val_hex)
#third column is P(Y==1)
GBM_valpredictions <- as.data.frame(GBM_valpred$pred)[,3]
labels <- as.data.frame(val_hex[,y])[,1]

#AUC expected
cvAUC::AUC(predictions = GBM_valpredictions, labels = labels) #0.9027938
###############
###########################################################################
# Check how each learner did so you can further tune 
##########################################################################

L <- length(GBM_learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(GBM_valpred$basepred)[,l], labels = labels)) 
data.frame(learner, auc)

########################################
# Generate predictions on the test set:
#######################################
GBM_pred <- predict(GBM_customfit, test_hex)
GBM_predictions <- as.data.frame(GBM_pred$pred)[,3]

#Creating a submission frame
submitter(hutest$mbr_id,GBM_predictions,"husubmission-ensembleGBM-12para") #Public leadboard score is 0.897
########################################################################################################




























































