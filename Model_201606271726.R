# 6/27: my 30 variables (set this up,and running overnight)
rm(list = ls())
library(h2o)
h2o.removeAll() # Clean slate - just in case the cluster was already running
library(h2oEnsemble)  # This will load the `h2o` R package as well
library(cvAUC) # cross validation AUC
localH2O = h2o.init(ip = '172.16.14.233', port = 54321,strict_version_check= FALSE)

# Load into h2o
attach("/hu/input/hu.RData")
hutrain_hex = as.h2o(hutrain)
hutest_hex = as.h2o(hutest)
y = "hu_01"
hutrain_hex[,y] <- as.factor(hutrain_hex[,y])
x <- hutrain_hex[,-c(141:145)]
x <- setdiff(names(x), y)
#str(x)
#str(y)

#####################################################
# Splitting into Training set and Validation set for model building
#####################################################
trainrows =nrow(hutrain)/3 # 2/3 train and 1/3 validation
train = subset(hutrain,mbr_id>=trainrows )
val = subset(hutrain,mbr_id<trainrows)
#####################################################
#Initialize h2o
#####################################################
# Load into h2o
train_hex = as.h2o(train)
val_hex = as.h2o(val)
test_hex = as.h2o(hutest)
## For binary classification, the response should be encoded as factor (also known as the enum type in Java).
train_hex[,y] <- as.factor(train_hex[,y])  
####################################################
#Running Models
#GBM: Gradient Boosted Models. For information on the GBM algorithm 

x30=c(
  "pri_cst",
  "loh_prspct_qt_p",
  "mcare_prspct_med_risk_qt_p",
  "rx_cncur_med_risk_qt_p",
  "rx_prspct_ttl_risk_qt",
  "cg_2014",
  "dx_prspct_med_risk_qt",
  "cms_hcc_130_ct",
  "hosp_day_ct_pri",
  "cops2_qt",
  "rx_prspct_ttl_risk_qt_p",
  "loh_prspct_qt",
  "mcare_prspct_med_risk_qt",
  "ethnct_ds_tx",
  "rx_inpat_prspct_ttl_risk_qt_p",
  "mcare_cncur_med_risk_qt_p",
  "cops2_qt_p",
  "rx_cncur_med_risk_qt",
  "age_yr_nb",
  "mcare_cncur_med_risk_qt",
  "er_stay_ct_pri",
  "paneled",
  "admt_ct_pri",
  "rx_elig_mm_ct",
  "rx_inpat_prspct_ttl_risk_qt",
  "dx_cncur_med_risk_qt",
  "cg_2012",
  "ae_elig_mm_ct",
  "dx_dem_prspct_med_risk_qt",
  "new_enrlee_score_qt"
  
)

# modeling
#My learner
h2o.glm.1 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.2 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.3 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.randomForest.1 <- function(..., ntrees = 200, nbins = 50, seed = 214) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.randomForest.2 <- function(..., ntrees = 200, sample_rate = 0.75, seed = 214) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.3 <- function(..., ntrees = 200, sample_rate = 0.85, seed = 214) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)

h2o.randomForest.4 <- function(..., ntrees = 200, nbins = 50, balance_classes = TRUE, seed = 214) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, balance_classes = balance_classes, seed = seed)
h2o.gbm.1 <- function(..., ntrees = 100, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.gbm.2 <- function(..., ntrees = 100, nbins = 50, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.gbm.3 <- function(..., ntrees = 100, max_depth = 10, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.4 <- function(..., ntrees = 100, col_sample_rate = 0.8, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.5 <- function(..., ntrees = 100, col_sample_rate = 0.7, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.6 <- function(..., ntrees = 100, col_sample_rate = 0.6, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.7 <- function(..., ntrees = 100, balance_classes = TRUE, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, seed = seed)
h2o.gbm.8 <- function(..., ntrees = 100, max_depth = 3, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
#using default learning rate is 0.1, build 500 treees
h2o.gbm.12 <- function(..., ntrees = 500, nbins = 50, learn_rate=0.1, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, learn_rate =learn_rate, seed = seed)
h2o.gbm.13 <- function(..., ntrees = 500, max_depth = 10, learn_rate=0.1, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, learn_rate =learn_rate,seed = seed)
h2o.gbm.14 <- function(..., ntrees = 500, col_sample_rate = 0.8, learn_rate=0.1, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate,learn_rate =learn_rate, seed = seed)
h2o.gbm.15 <- function(..., ntrees = 500, col_sample_rate = 0.7,learn_rate=0.1, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate,learn_rate =learn_rate, seed = seed)
h2o.gbm.16 <- function(..., ntrees = 500, col_sample_rate = 0.6, learn_rate=0.1, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate,learn_rate =learn_rate, seed = seed)
h2o.gbm.17 <- function(..., ntrees = 500, balance_classes = TRUE, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, seed = seed)

h2o.deeplearning.1 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.2 <- function(..., hidden = c(200,200,200), activation = "Tanh", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.3 <- function(..., hidden = c(500,500), activation = "RectifierWithDropout", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.4 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, balance_classes = TRUE, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, balance_classes = balance_classes, seed = seed)
h2o.deeplearning.5 <- function(..., hidden = c(100,100,100), activation = "Rectifier", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.6 <- function(..., hidden = c(50,50), activation = "Rectifier", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.7 <- function(..., hidden = c(100,100), activation = "Rectifier", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)

###################################
GBM_learner <- c(
  "h2o.glm.wrapper" 
  ,"h2o.gbm.wrapper" 
  ,"h2o.gbm.2"
  ,"h2o.gbm.3"
  ,"h2o.gbm.4"
  ,"h2o.gbm.5"
  ,"h2o.gbm.6"
  ,"h2o.gbm.7"
  ,"h2o.gbm.12"
  ,"h2o.gbm.13"
  ,"h2o.gbm.14"
  ,"h2o.gbm.15"
  ,"h2o.gbm.16"
)
metalearner <- "h2o.glm.wrapper"

GBM_customfit <- h2o.ensemble(x = x30, y = y,
                              training_frame = train_hex,
                              family='binomial',
                              learner = GBM_learner,
                              metalearner =metalearner,
                              cvControl = list(V = 5))
GBM_customfit30=GBM_customfit
#######################
#Check on Validation datasets
GBM_valpred <- predict(GBM_customfit, val_hex)
#third column is P(Y==1)
GBM_valpredictions <- as.data.frame(GBM_valpred$pred)[,3]
labels <- as.data.frame(val_hex[,y])[,1]

#AUC expected
cvAUC::AUC(predictions = GBM_valpredictions, labels = labels) #0.8981829
###############
###########################################################################
# Check how each learner did so you can further tune 
##########################################################################

L <- length(GBM_learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(GBM_valpred$basepred)[,l], labels = labels)) 


#GBM_learner       auc

########################################
# Generate predictions on the test set:
#######################################
GBM_pred <- predict(GBM_customfit, test_hex)
GBM_predictions <- as.data.frame(GBM_pred$pred)[,3]

#Creating a submission frame
submitter <- function(id,predictions,filename)
{ 
  submission<-cbind(id,predictions)
  colnames(submission) <- c("mbr_id", "prediction")
  submission <- as.data.frame(submission)
  #add your nuid 
  filename = paste0("/hu/output/l103295",filename,"l103295.csv")
  write.csv(submission, filename,row.names = FALSE)
}
submitter(hutest$mbr_id,GBM_predictions,"husubmission-enGBM_500tree-30para") #Public leadboard score is 0.8907
###########################################################################################################################################################

# 6/28: my 14 variables: original 12 plus emplaned and cg_2013

####################################################
#Running Models
#GBM: Gradient Boosted Models. For information on the GBM algorithm 

x14 = c("pri_cst","ethnct_ds_tx","dx_prspct_med_risk_qt","rx_prspct_ttl_risk_qt","mcare_prspct_med_risk_qt","loh_prspct_qt","cms_hcc_130_ct",
       "rx_inpat_prspct_ttl_risk_qt","cg_2014", "cops2_qt","rx_prspct_ttl_risk_qt_p","mcare_prspct_med_risk_qt_p","cg_2012","paneled")

# modeling
#My learner
h2o.glm.1 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.2 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.3 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.randomForest.1 <- function(..., ntrees = 200, nbins = 50, seed = 214) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.randomForest.2 <- function(..., ntrees = 200, sample_rate = 0.75, seed = 214) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.3 <- function(..., ntrees = 200, sample_rate = 0.85, seed = 214) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)

h2o.randomForest.4 <- function(..., ntrees = 200, nbins = 50, balance_classes = TRUE, seed = 214) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, balance_classes = balance_classes, seed = seed)
h2o.gbm.1 <- function(..., ntrees = 100, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.gbm.2 <- function(..., ntrees = 100, nbins = 50, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.gbm.3 <- function(..., ntrees = 100, max_depth = 10, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.4 <- function(..., ntrees = 100, col_sample_rate = 0.8, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.5 <- function(..., ntrees = 100, col_sample_rate = 0.7, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.6 <- function(..., ntrees = 100, col_sample_rate = 0.6, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.7 <- function(..., ntrees = 100, balance_classes = TRUE, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, seed = seed)
h2o.gbm.8 <- function(..., ntrees = 100, max_depth = 3, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
#using default learning rate is 0.1, build 500 treees
h2o.gbm.12 <- function(..., ntrees = 500, nbins = 50, learn_rate=0.1, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, learn_rate =learn_rate, seed = seed)
h2o.gbm.13 <- function(..., ntrees = 500, max_depth = 10, learn_rate=0.1, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, learn_rate =learn_rate,seed = seed)
h2o.gbm.14 <- function(..., ntrees = 500, col_sample_rate = 0.8, learn_rate=0.1, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate,learn_rate =learn_rate, seed = seed)
h2o.gbm.15 <- function(..., ntrees = 500, col_sample_rate = 0.7,learn_rate=0.1, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate,learn_rate =learn_rate, seed = seed)
h2o.gbm.16 <- function(..., ntrees = 500, col_sample_rate = 0.6, learn_rate=0.1, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate,learn_rate =learn_rate, seed = seed)

h2o.deeplearning.1 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.2 <- function(..., hidden = c(200,200,200), activation = "Tanh", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.3 <- function(..., hidden = c(500,500), activation = "RectifierWithDropout", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.4 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, balance_classes = TRUE, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, balance_classes = balance_classes, seed = seed)
h2o.deeplearning.5 <- function(..., hidden = c(100,100,100), activation = "Rectifier", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.6 <- function(..., hidden = c(50,50), activation = "Rectifier", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.7 <- function(..., hidden = c(100,100), activation = "Rectifier", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)

###################################
GBM_learner <- c(
  "h2o.glm.wrapper" 
  ,"h2o.gbm.wrapper" 
  ,"h2o.gbm.2"
  ,"h2o.gbm.3"
  ,"h2o.gbm.4"
  ,"h2o.gbm.5"
  ,"h2o.gbm.6"
  ,"h2o.gbm.7"
  ,"h2o.gbm.12"
  ,"h2o.gbm.13"
  ,"h2o.gbm.14"
  ,"h2o.gbm.15"
  ,"h2o.gbm.16"
)
metalearner <- "h2o.glm.wrapper"

GBM_customfit <- h2o.ensemble(x = x14, y = y,
                              training_frame = train_hex,
                              family='binomial',
                              learner = GBM_learner,
                              metalearner =metalearner,
                              cvControl = list(V = 5))

GBM_customfit14=GBM_customfit
#######################
#Check on Validation datasets
GBM_valpred <- predict(GBM_customfit, val_hex)
#third column is P(Y==1)
GBM_valpredictions <- as.data.frame(GBM_valpred$pred)[,3]
labels <- as.data.frame(val_hex[,y])[,1]

#AUC expected
cvAUC::AUC(predictions = GBM_valpredictions, labels = labels)  #0.8954066
###############
###########################################################################
# Check how each learner did so you can further tune 
##########################################################################

L <- length(GBM_learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(GBM_valpred$basepred)[,l], labels = labels)) 
data.frame(GBM_learner, auc)


#GBM_learner       auc
#1  h2o.glm.wrapper 0.8770189
#2  h2o.gbm.wrapper 0.8994659
#3        h2o.gbm.2 0.9008426
#4        h2o.gbm.3 0.8973936
#5        h2o.gbm.4 0.9010358
#6        h2o.gbm.5 0.9010901
#7        h2o.gbm.6 0.9014722
#8        h2o.gbm.7 0.8940900
#9       h2o.gbm.12 0.9014736
#10      h2o.gbm.13 0.8638689
#11      h2o.gbm.14 0.9019046
#12      h2o.gbm.15 0.9025039
#13      h2o.gbm.16 0.9027357
########################################
# Generate predictions on the test set:
#######################################
GBM_pred <- predict(GBM_customfit, test_hex)
GBM_predictions <- as.data.frame(GBM_pred$pred)[,3]

#Creating a submission frame
submitter <- function(id,predictions,filename)
{ 
  submission<-cbind(id,predictions)
  colnames(submission) <- c("mbr_id", "prediction")
  submission <- as.data.frame(submission)
  #add your nuid 
  filename = paste0("/hu/output/l103295",filename,"l103295.csv")
  write.csv(submission, filename,row.names = FALSE)
}
submitter(hutest$mbr_id,GBM_predictions,"husubmission-enGBM_500tree-14para") #Public leadboard score is 0.888
###########################################################################################################################################################


