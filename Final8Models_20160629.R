#6/29: Final Model Testing:
# With NA, no ambiguous factor like (X or ?)
#Model#1: Using GBM Learner 1-8 with 12 var
#Model#2: Using GBM Learner 2-6 with 12 var: public Leaderboar: 0.895
#Model#3: Using GBM Learner 1-8 with 9 var
#Model#4: Using GBM Learner 2-6 with 9 var public Leaderboar: 0.894
# Removing NA, and no ambiguous factor like (X or ?)
#Model#5: Using GBM Learner 1-8 with 12 var
#[1] 0.904697
#GBM_learner       auc
#1   h2o.gbm.1 0.9038390
#2   h2o.gbm.2 0.9034395
#3   h2o.gbm.3 0.9032725
#4   h2o.gbm.4 0.9037514
#5   h2o.gbm.5 0.9033722
#6   h2o.gbm.6 0.9037109
#7   h2o.gbm.7 0.8993304
#8   h2o.gbm.8 0.9018360
#Model#6: Using GBM Learner 2-6 with 12 var: PL: 0.834
#1] 0.904791
#GBM_learner       auc
#1   h2o.gbm.2 0.9034395
#2   h2o.gbm.3 0.9025846
#3   h2o.gbm.4 0.9037514
#4   h2o.gbm.5 0.9033722
#5   h2o.gbm.6 0.9037109
#Model#7: Using GBM Learner 1-8 with 9 var
#[1] 0.9045333
#GBM_learner       auc
#1   h2o.gbm.1 0.9027580
#2   h2o.gbm.2 0.9034006
#3   h2o.gbm.3 0.9023455
#4   h2o.gbm.4 0.9032333
#5   h2o.gbm.5 0.9036421
#6   h2o.gbm.6 0.9031458
#7   h2o.gbm.7 0.8987177
#8   h2o.gbm.8 0.9019161
#Model#8: Using GBM Learner 2-6 with 9 var
#[1] 0.9042494
#GBM_learner       auc
#1   h2o.gbm.2 0.9034006
#2   h2o.gbm.3 0.9022764
#3   h2o.gbm.4 0.9032333
#4   h2o.gbm.5 0.9036421
#5   h2o.gbm.6 0.9031458
##########################################################

rm(list = ls())
library(h2o)
h2o.removeAll() # Clean slate - just in case the cluster was already running
library(h2oEnsemble)  # This will load the `h2o` R package as well
library(cvAUC) # cross validation AUC
localH2O = h2o.init(ip = '172.16.14.233', port = 54321,strict_version_check= FALSE)

# Load into h2o
attach("/hu/input/hu.RData")
library(sqldf)
hutrain_13var<- sqldf('select mbr_id,
                      pri_cst,
                      ethnct_ds_tx,
                      dx_prspct_med_risk_qt,
                      rx_prspct_ttl_risk_qt,
                      mcare_prspct_med_risk_qt,
                      loh_prspct_qt,
                      cms_hcc_130_ct,
                      rx_inpat_prspct_ttl_risk_qt,
                      cg_2014,
                      cops2_qt,
                      rx_prspct_ttl_risk_qt_p,
                      mcare_prspct_med_risk_qt_p,
                      hu_01
                      from hutrain')


str(hutrain)
str(hutrain_13var)
hutrain_13var$ethnct_ds_tx= factor(ifelse(hutrain_13var$ethnct_ds_tx  == "?", NA, as.character(hutrain_13var$ethnct_ds_tx)))
hutrain_13var$cg_2014 = factor(ifelse(hutrain_13var$cg_2014  == "X", NA, as.character(hutrain_13var$cg_2014)))
hutrain_13var$cms_hcc_130_ct=as.factor(hutrain_13var$cms_hcc_130_ct)
hutrain_13var$hu_01=as.factor(hutrain_13var$hu_01)
summary(hutrain_13var)

#cleaning my test dataset
hutest_12var<- sqldf('select mbr_id,
                     pri_cst,
                     ethnct_ds_tx,
                     dx_prspct_med_risk_qt,
                     rx_prspct_ttl_risk_qt,
                     mcare_prspct_med_risk_qt,
                     loh_prspct_qt,
                     cms_hcc_130_ct,
                     rx_inpat_prspct_ttl_risk_qt,
                     cg_2014,
                     cops2_qt,
                     rx_prspct_ttl_risk_qt_p,
                     mcare_prspct_med_risk_qt_p
                     from hutest')


#str(hutest_12var)
hutest_12var$ethnct_ds_tx= factor(ifelse(hutest_12var$ethnct_ds_tx  == "?", NA, as.character(hutest_12var$ethnct_ds_tx)))
hutest_12var$cg_2014 = factor(ifelse(hutest_12var$cg_2014  == "X", NA, as.character(hutest_12var$cg_2014)))
hutest_12var$cms_hcc_130_ct=as.factor(hutest_12var$cms_hcc_130_ct)
#summary(hutest_12var)

##########################################
hutrain_hex = as.h2o(hutrain_13var)
hutest_hex = as.h2o(hutest_12var)
y = "hu_01"
hutrain_hex[,y] <- as.factor(hutrain_hex[,y])
x <- hutrain_hex[,-c(14)]
x <- setdiff(names(x), y)
str(x)
#str(y)

#####################################################
# Splitting into Training set and Validation set for model building
#####################################################
trainrows =nrow(hutrain_13var)/3 # 2/3 train and 1/3 validation
train = subset(hutrain_13var,mbr_id>=trainrows )
val = subset(hutrain_13var,mbr_id<trainrows)
#####################################################
#Initialize h2o
#####################################################
# Load into h2o
train_hex = as.h2o(train)
summary(train_hex )
val_hex = as.h2o(val)
test_hex = as.h2o(hutest_12var)

#####################################################################
# Random Forest - I do this to get the important variables
#####################################################################
#hu.rf = h2o.randomForest(y = y, x = x, training_frame = train_hex)

#Lets look at variables since we will get penalized if used all variables
impvariables = h2o.varimp(hu.rf)
View(impvariables)

## For binary classification, the response should be encoded as factor (also known as the enum type in Java).
train_hex[,y] <- as.factor(train_hex[,y])  
x12 = c("pri_cst","ethnct_ds_tx","dx_prspct_med_risk_qt","rx_prspct_ttl_risk_qt","mcare_prspct_med_risk_qt","loh_prspct_qt","cms_hcc_130_ct",
        "rx_inpat_prspct_ttl_risk_qt","cg_2014","cops2_qt","rx_prspct_ttl_risk_qt_p","mcare_prspct_med_risk_qt_p")
#My learner
# modeling
#My learner
#h2o.glm.1 <- function(..., alpha = 0.0, family="binomial") h2o.glm.wrapper(..., alpha = alpha, family = family)
#h2o.glm.2 <- function(..., alpha = 0.5, family="binomial") h2o.glm.wrapper(..., alpha = alpha, family = family)
#h2o.glm.3 <- function(..., alpha = 1.0, family="binomial") h2o.glm.wrapper(..., alpha = alpha, family = family)
#adding new GLM models
#h2o.glm.4 <- function(..., alpha = 0.5, family="binomial", max_iterations = 500, missing_values_handling = "MeanImputation") h2o.glm.wrapper(..., alpha = alpha,missing_values_handling = missing_values_handling,max_iterations=max_iterations, family = family)
#h2o.glm.5 <- function(..., alpha = 0.5, family="binomial", max_iterations = 500, missing_values_handling = "Skip")h2o.glm.wrapper(..., alpha = alpha,missing_values_handling = missing_values_handling, max_iterations=max_iterations, family = family)
#h2o.glm.6 <- function(..., alpha = 0.5, family="binomial", max_iterations = 500, solver = "IRLSM",missing_values_handling = "Skip")h2o.glm.wrapper(..., alpha = alpha, missing_values_handling = missing_values_handling, max_iterations=max_iterations, solver=solver, family = family)
#h2o.glm.7 <- function(..., alpha = 0.5, family="binomial", max_iterations = 500, solver = "IRLSM",missing_values_handling = "MeanImputation")h2o.glm.wrapper(..., alpha = alpha, missing_values_handling = missing_values_handling, max_iterations=max_iterations, solver=solver, family = family)
h2o.glm.8 <- function(..., alpha = 0.5, family="binomial", max_iterations = 500, solver = "L_BFGS",missing_values_handling = "Skip")h2o.glm.wrapper(..., alpha = alpha, missing_values_handling = missing_values_handling, max_iterations=max_iterations, solver=solver, family = family)
#h2o.glm.9 <- function(..., alpha = 0.5, family="binomial", max_iterations = 500, solver = "L_BFGS",missing_values_handling = "MeanImputation")h2o.glm.wrapper(..., alpha = alpha, missing_values_handling = missing_values_handling, max_iterations=max_iterations, solver=solver, family = family)
#h2o.glm.10 <- function(..., alpha = 0.5, family="binomial", max_iterations = 500, solver = "L_BFGS",missing_values_handling = "MeanImputation", standardize = TRUE)h2o.glm.wrapper(..., alpha = alpha, missing_values_handling = missing_values_handling, max_iterations=max_iterations, solver=solver, family = family, standardize = standardize)
#RF
#h2o.randomForest.1 <- function(..., ntrees = 200, nbins = 50, seed = 214) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
#h2o.randomForest.2 <- function(..., ntrees = 200, sample_rate = 0.75, seed = 214) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
#h2o.randomForest.3 <- function(..., ntrees = 200, sample_rate = 0.85, seed = 214) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
#h2o.randomForest.4 <- function(..., ntrees = 200, nbins = 50, balance_classes = TRUE, seed = 214) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, balance_classes = balance_classes, seed = seed)
h2o.gbm.1 <- function(..., ntrees = 100, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.gbm.2 <- function(..., ntrees = 100, nbins = 50, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.gbm.3 <- function(..., ntrees = 100, max_depth = 10, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.4 <- function(..., ntrees = 100, col_sample_rate = 0.8, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.5 <- function(..., ntrees = 100, col_sample_rate = 0.7, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.6 <- function(..., ntrees = 100, col_sample_rate = 0.6, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.7 <- function(..., ntrees = 100, balance_classes = TRUE, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, seed = seed)
h2o.gbm.8 <- function(..., ntrees = 100, max_depth = 3, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
#Testing learning_rate
h2o.gbm.9 <- function(..., ntrees = 100, balance_classes = TRUE, learn_rate=0.8, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, learn_rate =learn_rate, seed = seed)
h2o.gbm.10 <- function(..., ntrees = 100, balance_classes = TRUE, learn_rate=0.2, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, learn_rate =learn_rate, seed = seed)
h2o.gbm.11 <- function(..., ntrees = 100, balance_classes = TRUE, learn_rate=0.5, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, learn_rate =learn_rate, seed = seed)
#using learning_rate on gbm.2 to gbm.6 (6/28, adding new gbm learners)
#h2o.gbm.12 <- function(..., ntrees = 100, nbins = 50, learn_rate=0.2, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, learn_rate =learn_rate, seed = seed)
#h2o.gbm.13 <- function(..., ntrees = 100, max_depth = 30, learn_rate=0.2, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, learn_rate =learn_rate,seed = seed)
#h2o.gbm.14 <- function(..., ntrees = 100, col_sample_rate = 0.8, learn_rate=0.2, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate,learn_rate =learn_rate, seed = seed)
#h2o.gbm.15 <- function(..., ntrees = 100, col_sample_rate = 0.7,learn_rate=0.2, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate,learn_rate =learn_rate, seed = seed)
#h2o.gbm.16 <- function(..., ntrees = 100, col_sample_rate = 0.6, learn_rate=0.2, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate,learn_rate =learn_rate, seed = seed)
#adding more gmb model
#h2o.gbm.17 <- function(..., ntrees = 200, max_depth = 10, learn_rate=0.3, col_sample_rate=0.7,stopping_round=2, stopping_tolerance=0.01, score_eaach_iteration=T,  seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, learn_rate=learn_rate, col_sample_rate=col_sample_rate,stopping_round=stopping_round, stopping_tolerance=stopping_tolerance, score_eaach_iteration =score_eaach_iteration, seed = seed)
#h2o.gbm.18 <- function(..., ntrees = 100, max_depth = 30, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.24 <- function(..., ntrees = 100, col_sample_rate = 0.8, max_depth = 10, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, max_depth = max_depth, seed = seed)
h2o.gbm.25 <- function(..., ntrees = 100, col_sample_rate = 0.7,max_depth = 10, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, max_depth = max_depth, seed = seed)
h2o.gbm.26 <- function(..., ntrees = 100, col_sample_rate = 0.6,max_depth = 10, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, max_depth = max_depth, seed = seed)
h2o.gbm.28 <- function(..., ntrees = 100, col_sample_rate = 0.8, max_depth = 3, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, max_depth = max_depth, seed = seed)
h2o.gbm.34 <- function(..., ntrees = 100, col_sample_rate = 0.8, max_depth = 3, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, max_depth = max_depth, seed = seed)
h2o.gbm.35 <- function(..., ntrees = 100, col_sample_rate = 0.7,max_depth = 3, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, max_depth = max_depth, seed = seed)
h2o.gbm.36 <- function(..., ntrees = 100, col_sample_rate = 0.6,max_depth = 3, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, max_depth = max_depth, seed = seed)

h2o.deeplearning.1 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.2 <- function(..., hidden = c(200,200,200), activation = "Tanh", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.3 <- function(..., hidden = c(500,500), activation = "RectifierWithDropout", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.4 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, balance_classes = TRUE, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, balance_classes = balance_classes, seed = seed)
h2o.deeplearning.5 <- function(..., hidden = c(100,100,100), activation = "Rectifier", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.6 <- function(..., hidden = c(50,50), activation = "Rectifier", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.7 <- function(..., hidden = c(100,100), activation = "Rectifier", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)

###################################
GBM_learner <- c(
  "h2o.gbm.1"
  ,"h2o.gbm.2"
  ,"h2o.gbm.3"
  ,"h2o.gbm.4"
  ,"h2o.gbm.5"
  ,"h2o.gbm.6"
  ,"h2o.gbm.7"
  ,"h2o.gbm.8"
  #,"h2o.gbm.24"
  #  ,"h2o.gbm.25"
  #  ,"h2o.gbm.26"
  #  ,"h2o.gbm.28"
  #  ,"h2o.gbm.34"
  #  ,"h2o.gbm.35"
  #  ,"h2o.gbm.36"
)

metalearner <- "h2o.glm.wrapper"

GBM_customfit <- h2o.ensemble(x = x12, y = y,
                              training_frame = train_hex,
                              family='binomial',
                              learner = GBM_learner,
                              metalearner =metalearner,
                              cvControl = list(V = 5))

#######################
#Check on Validation datasets
GBM_valpred <- predict(GBM_customfit, val_hex)
#third column is P(Y==1)
GBM_valpredictions <- as.data.frame(GBM_valpred$pred)[,3]
labels <- as.data.frame(val_hex[,y])[,1]

#AUC expected
cvAUC::AUC(predictions = GBM_valpredictions, labels = labels)
###############
###########################################################################
# Check how each learner did so you can further tune 
##########################################################################

L <- length(GBM_learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(GBM_valpred$basepred)[,l], labels = labels)) 
data.frame(GBM_learner, auc)
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
submitter(hutest$mbr_id,GBM_predictions,"husubmission_model1") #Public leadboard score is 0.
###########################################################################################################
#Model 2
GBM_learner <- c(
  #  "h2o.gbm.1"
  "h2o.gbm.2"
  ,"h2o.gbm.3"
  ,"h2o.gbm.4"
  ,"h2o.gbm.5"
  ,"h2o.gbm.6"
  # ,"h2o.gbm.7"
  # ,"h2o.gbm.8"
  #,"h2o.gbm.24"
  #  ,"h2o.gbm.25"
  #  ,"h2o.gbm.26"
  #  ,"h2o.gbm.28"
  #  ,"h2o.gbm.34"
  #  ,"h2o.gbm.35"
  #  ,"h2o.gbm.36"
)

metalearner <- "h2o.glm.wrapper"

GBM_customfit <- h2o.ensemble(x = x12, y = y,
                              training_frame = train_hex,
                              family='binomial',
                              learner = GBM_learner,
                              metalearner =metalearner,
                              cvControl = list(V = 5))

#######################
#Check on Validation datasets
GBM_valpred <- predict(GBM_customfit, val_hex)
#third column is P(Y==1)
GBM_valpredictions <- as.data.frame(GBM_valpred$pred)[,3]
labels <- as.data.frame(val_hex[,y])[,1]

#AUC expected
cvAUC::AUC(predictions = GBM_valpredictions, labels = labels)
###############
###########################################################################
# Check how each learner did so you can further tune 
##########################################################################

L <- length(GBM_learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(GBM_valpred$basepred)[,l], labels = labels)) 
data.frame(GBM_learner, auc)
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
submitter(hutest$mbr_id,GBM_predictions,"husubmission_model2") #Public leadboard score is 0.
#################
#Model 3
x9= c("pri_cst","ethnct_ds_tx","dx_prspct_med_risk_qt","rx_prspct_ttl_risk_qt","mcare_prspct_med_risk_qt","loh_prspct_qt",
      "rx_inpat_prspct_ttl_risk_qt","cops2_qt","mcare_prspct_med_risk_qt_p")

GBM_learner <- c(
  "h2o.gbm.1"
  ,"h2o.gbm.2"
  ,"h2o.gbm.3"
  ,"h2o.gbm.4"
  ,"h2o.gbm.5"
  ,"h2o.gbm.6"
  ,"h2o.gbm.7"
  ,"h2o.gbm.8"
  #,"h2o.gbm.24"
  #  ,"h2o.gbm.25"
  #  ,"h2o.gbm.26"
  #  ,"h2o.gbm.28"
  #  ,"h2o.gbm.34"
  #  ,"h2o.gbm.35"
  #  ,"h2o.gbm.36"
)

metalearner <- "h2o.glm.wrapper"

GBM_customfit <- h2o.ensemble(x = x9, y = y,
                              training_frame = train_hex,
                              family='binomial',
                              learner = GBM_learner,
                              metalearner =metalearner,
                              cvControl = list(V = 5))

#######################
#Check on Validation datasets
GBM_valpred <- predict(GBM_customfit, val_hex)
#third column is P(Y==1)
GBM_valpredictions <- as.data.frame(GBM_valpred$pred)[,3]
labels <- as.data.frame(val_hex[,y])[,1]

#AUC expected
cvAUC::AUC(predictions = GBM_valpredictions, labels = labels)
###############
###########################################################################
# Check how each learner did so you can further tune 
##########################################################################

L <- length(GBM_learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(GBM_valpred$basepred)[,l], labels = labels)) 
data.frame(GBM_learner, auc)
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
submitter(hutest$mbr_id,GBM_predictions,"husubmission_model3") #Public leadboard score is 0.
###########################################################################################################
#Model 4
GBM_learner <- c(
  #  "h2o.gbm.1"
  "h2o.gbm.2"
  ,"h2o.gbm.3"
  ,"h2o.gbm.4"
  ,"h2o.gbm.5"
  ,"h2o.gbm.6"
  # ,"h2o.gbm.7"
  # ,"h2o.gbm.8"
  #,"h2o.gbm.24"
  #  ,"h2o.gbm.25"
  #  ,"h2o.gbm.26"
  #  ,"h2o.gbm.28"
  #  ,"h2o.gbm.34"
  #  ,"h2o.gbm.35"
  #  ,"h2o.gbm.36"
)

metalearner <- "h2o.glm.wrapper"

GBM_customfit <- h2o.ensemble(x = x9, y = y,
                              training_frame = train_hex,
                              family='binomial',
                              learner = GBM_learner,
                              metalearner =metalearner,
                              cvControl = list(V = 5))

#######################
#Check on Validation datasets
GBM_valpred <- predict(GBM_customfit, val_hex)
#third column is P(Y==1)
GBM_valpredictions <- as.data.frame(GBM_valpred$pred)[,3]
labels <- as.data.frame(val_hex[,y])[,1]

#AUC expected
cvAUC::AUC(predictions = GBM_valpredictions, labels = labels)
###############
###########################################################################
# Check how each learner did so you can further tune 
##########################################################################

L <- length(GBM_learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(GBM_valpred$basepred)[,l], labels = labels)) 
data.frame(GBM_learner, auc)
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
submitter(hutest$mbr_id,GBM_predictions,"husubmission_model4") #Public leadboard score is 0.		




############################################################
#################################################################
#Part II: Removing NAs

hutrain_13var<- sqldf('select mbr_id,
                      pri_cst,
                      ethnct_ds_tx,
                      dx_prspct_med_risk_qt,
                      rx_prspct_ttl_risk_qt,
                      mcare_prspct_med_risk_qt,
                      loh_prspct_qt,
                      cms_hcc_130_ct,
                      rx_inpat_prspct_ttl_risk_qt,
                      cg_2014,
                      cops2_qt,
                      rx_prspct_ttl_risk_qt_p,
                      mcare_prspct_med_risk_qt_p,
                      hu_01
                      from hutrain')


#str(hutrain)
#str(hutrain_13var)
hutrain_13var$ethnct_ds_tx= factor(ifelse(hutrain_13var$ethnct_ds_tx  == "?", NA, as.character(hutrain_13var$ethnct_ds_tx)))
hutrain_13var$cg_2014 = factor(ifelse(hutrain_13var$cg_2014  == "X", NA, as.character(hutrain_13var$cg_2014)))
hutrain_13var$cms_hcc_130_ct=as.factor(hutrain_13var$cms_hcc_130_ct)
hutrain_13var$hu_01=as.factor(hutrain_13var$hu_01)
summary(hutrain_13var)
complete.cases(hutrain_13var)
hutrain_13var<-hutrain_13var[complete.cases(hutrain_13var),]
str(hutrain_13var) # 820386 obs  withNA is 1000000, removing 179614 cases
hutrain_13var<-na.omit(hutrain_13var)
############################
#cleaning my test dataset
hutest_12var<- sqldf('select mbr_id,
                     pri_cst,
                     ethnct_ds_tx,
                     dx_prspct_med_risk_qt,
                     rx_prspct_ttl_risk_qt,
                     mcare_prspct_med_risk_qt,
                     loh_prspct_qt,
                     cms_hcc_130_ct,
                     rx_inpat_prspct_ttl_risk_qt,
                     cg_2014,
                     cops2_qt,
                     rx_prspct_ttl_risk_qt_p,
                     mcare_prspct_med_risk_qt_p
                     from hutest')


#str(hutest_12var)
hutest_12var$ethnct_ds_tx= factor(ifelse(hutest_12var$ethnct_ds_tx  == "?", NA, as.character(hutest_12var$ethnct_ds_tx)))
hutest_12var$cg_2014 = factor(ifelse(hutest_12var$cg_2014  == "X", NA, as.character(hutest_12var$cg_2014)))
hutest_12var$cms_hcc_130_ct=as.factor(hutest_12var$cms_hcc_130_ct)
#summary(hutest_12var)

##########################################
hutrain_hex = as.h2o(hutrain_13var)
hutest_hex = as.h2o(hutest_12var)
y = "hu_01"
hutrain_hex[,y] <- as.factor(hutrain_hex[,y])
x <- hutrain_hex[,-c(14)]
x <- setdiff(names(x), y)
str(x)
#str(y)

#####################################################
# Splitting into Training set and Validation set for model building
#####################################################
trainrows =nrow(hutrain_13var)/3 # 2/3 train and 1/3 validation
train = subset(hutrain_13var,mbr_id>=trainrows )
val = subset(hutrain_13var,mbr_id<trainrows)
#####################################################
#Initialize h2o
#####################################################
# Load into h2o
train_hex = as.h2o(train)
summary(train_hex )
val_hex = as.h2o(val)
test_hex = as.h2o(hutest_12var)

#####################################################################
# Random Forest - I do this to get the important variables
#####################################################################
#hu.rf = h2o.randomForest(y = y, x = x, training_frame = train_hex)

#Lets look at variables since we will get penalized if used all variables
impvariables = h2o.varimp(hu.rf)
View(impvariables)

## For binary classification, the response should be encoded as factor (also known as the enum type in Java).
train_hex[,y] <- as.factor(train_hex[,y])  
x12 = c("pri_cst","ethnct_ds_tx","dx_prspct_med_risk_qt","rx_prspct_ttl_risk_qt","mcare_prspct_med_risk_qt","loh_prspct_qt","cms_hcc_130_ct",
        "rx_inpat_prspct_ttl_risk_qt","cg_2014","cops2_qt","rx_prspct_ttl_risk_qt_p","mcare_prspct_med_risk_qt_p")
#My learner
# modeling
#My learner
#h2o.glm.1 <- function(..., alpha = 0.0, family="binomial") h2o.glm.wrapper(..., alpha = alpha, family = family)
#h2o.glm.2 <- function(..., alpha = 0.5, family="binomial") h2o.glm.wrapper(..., alpha = alpha, family = family)
#h2o.glm.3 <- function(..., alpha = 1.0, family="binomial") h2o.glm.wrapper(..., alpha = alpha, family = family)
#adding new GLM models
#h2o.glm.4 <- function(..., alpha = 0.5, family="binomial", max_iterations = 500, missing_values_handling = "MeanImputation") h2o.glm.wrapper(..., alpha = alpha,missing_values_handling = missing_values_handling,max_iterations=max_iterations, family = family)
#h2o.glm.5 <- function(..., alpha = 0.5, family="binomial", max_iterations = 500, missing_values_handling = "Skip")h2o.glm.wrapper(..., alpha = alpha,missing_values_handling = missing_values_handling, max_iterations=max_iterations, family = family)
#h2o.glm.6 <- function(..., alpha = 0.5, family="binomial", max_iterations = 500, solver = "IRLSM",missing_values_handling = "Skip")h2o.glm.wrapper(..., alpha = alpha, missing_values_handling = missing_values_handling, max_iterations=max_iterations, solver=solver, family = family)
#h2o.glm.7 <- function(..., alpha = 0.5, family="binomial", max_iterations = 500, solver = "IRLSM",missing_values_handling = "MeanImputation")h2o.glm.wrapper(..., alpha = alpha, missing_values_handling = missing_values_handling, max_iterations=max_iterations, solver=solver, family = family)
h2o.glm.8 <- function(..., alpha = 0.5, family="binomial", max_iterations = 500, solver = "L_BFGS",missing_values_handling = "Skip")h2o.glm.wrapper(..., alpha = alpha, missing_values_handling = missing_values_handling, max_iterations=max_iterations, solver=solver, family = family)
#h2o.glm.9 <- function(..., alpha = 0.5, family="binomial", max_iterations = 500, solver = "L_BFGS",missing_values_handling = "MeanImputation")h2o.glm.wrapper(..., alpha = alpha, missing_values_handling = missing_values_handling, max_iterations=max_iterations, solver=solver, family = family)
#h2o.glm.10 <- function(..., alpha = 0.5, family="binomial", max_iterations = 500, solver = "L_BFGS",missing_values_handling = "MeanImputation", standardize = TRUE)h2o.glm.wrapper(..., alpha = alpha, missing_values_handling = missing_values_handling, max_iterations=max_iterations, solver=solver, family = family, standardize = standardize)
#RF
#h2o.randomForest.1 <- function(..., ntrees = 200, nbins = 50, seed = 214) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
#h2o.randomForest.2 <- function(..., ntrees = 200, sample_rate = 0.75, seed = 214) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
#h2o.randomForest.3 <- function(..., ntrees = 200, sample_rate = 0.85, seed = 214) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
#h2o.randomForest.4 <- function(..., ntrees = 200, nbins = 50, balance_classes = TRUE, seed = 214) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, balance_classes = balance_classes, seed = seed)
h2o.gbm.1 <- function(..., ntrees = 100, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.gbm.2 <- function(..., ntrees = 100, nbins = 50, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.gbm.3 <- function(..., ntrees = 100, max_depth = 10, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.4 <- function(..., ntrees = 100, col_sample_rate = 0.8, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.5 <- function(..., ntrees = 100, col_sample_rate = 0.7, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.6 <- function(..., ntrees = 100, col_sample_rate = 0.6, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.7 <- function(..., ntrees = 100, balance_classes = TRUE, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, seed = seed)
h2o.gbm.8 <- function(..., ntrees = 100, max_depth = 3, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
#Testing learning_rate
h2o.gbm.9 <- function(..., ntrees = 100, balance_classes = TRUE, learn_rate=0.8, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, learn_rate =learn_rate, seed = seed)
h2o.gbm.10 <- function(..., ntrees = 100, balance_classes = TRUE, learn_rate=0.2, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, learn_rate =learn_rate, seed = seed)
h2o.gbm.11 <- function(..., ntrees = 100, balance_classes = TRUE, learn_rate=0.5, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, learn_rate =learn_rate, seed = seed)
#using learning_rate on gbm.2 to gbm.6 (6/28, adding new gbm learners)
#h2o.gbm.12 <- function(..., ntrees = 100, nbins = 50, learn_rate=0.2, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, learn_rate =learn_rate, seed = seed)
#h2o.gbm.13 <- function(..., ntrees = 100, max_depth = 30, learn_rate=0.2, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, learn_rate =learn_rate,seed = seed)
#h2o.gbm.14 <- function(..., ntrees = 100, col_sample_rate = 0.8, learn_rate=0.2, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate,learn_rate =learn_rate, seed = seed)
#h2o.gbm.15 <- function(..., ntrees = 100, col_sample_rate = 0.7,learn_rate=0.2, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate,learn_rate =learn_rate, seed = seed)
#h2o.gbm.16 <- function(..., ntrees = 100, col_sample_rate = 0.6, learn_rate=0.2, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate,learn_rate =learn_rate, seed = seed)
#adding more gmb model
#h2o.gbm.17 <- function(..., ntrees = 200, max_depth = 10, learn_rate=0.3, col_sample_rate=0.7,stopping_round=2, stopping_tolerance=0.01, score_eaach_iteration=T,  seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, learn_rate=learn_rate, col_sample_rate=col_sample_rate,stopping_round=stopping_round, stopping_tolerance=stopping_tolerance, score_eaach_iteration =score_eaach_iteration, seed = seed)
#h2o.gbm.18 <- function(..., ntrees = 100, max_depth = 30, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.24 <- function(..., ntrees = 100, col_sample_rate = 0.8, max_depth = 10, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, max_depth = max_depth, seed = seed)
h2o.gbm.25 <- function(..., ntrees = 100, col_sample_rate = 0.7,max_depth = 10, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, max_depth = max_depth, seed = seed)
h2o.gbm.26 <- function(..., ntrees = 100, col_sample_rate = 0.6,max_depth = 10, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, max_depth = max_depth, seed = seed)
h2o.gbm.28 <- function(..., ntrees = 100, col_sample_rate = 0.8, max_depth = 3, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, max_depth = max_depth, seed = seed)
h2o.gbm.34 <- function(..., ntrees = 100, col_sample_rate = 0.8, max_depth = 3, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, max_depth = max_depth, seed = seed)
h2o.gbm.35 <- function(..., ntrees = 100, col_sample_rate = 0.7,max_depth = 3, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, max_depth = max_depth, seed = seed)
h2o.gbm.36 <- function(..., ntrees = 100, col_sample_rate = 0.6,max_depth = 3, seed = 214) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, max_depth = max_depth, seed = seed)

h2o.deeplearning.1 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.2 <- function(..., hidden = c(200,200,200), activation = "Tanh", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.3 <- function(..., hidden = c(500,500), activation = "RectifierWithDropout", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.4 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, balance_classes = TRUE, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, balance_classes = balance_classes, seed = seed)
h2o.deeplearning.5 <- function(..., hidden = c(100,100,100), activation = "Rectifier", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.6 <- function(..., hidden = c(50,50), activation = "Rectifier", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.7 <- function(..., hidden = c(100,100), activation = "Rectifier", epochs = 50, seed = 214)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)

###################################
#Model#5
GBM_learner <- c(
  "h2o.gbm.1"
  ,"h2o.gbm.2"
  ,"h2o.gbm.3"
  ,"h2o.gbm.4"
  ,"h2o.gbm.5"
  ,"h2o.gbm.6"
  ,"h2o.gbm.7"
  ,"h2o.gbm.8"
  #,"h2o.gbm.24"
  #  ,"h2o.gbm.25"
  #  ,"h2o.gbm.26"
  #  ,"h2o.gbm.28"
  #  ,"h2o.gbm.34"
  #  ,"h2o.gbm.35"
  #  ,"h2o.gbm.36"
)

metalearner <- "h2o.glm.wrapper"

GBM_customfit <- h2o.ensemble(x = x12, y = y,
                              training_frame = train_hex,
                              family='binomial',
                              learner = GBM_learner,
                              metalearner =metalearner,
                              cvControl = list(V = 5))

#######################
#Check on Validation datasets
GBM_valpred <- predict(GBM_customfit, val_hex)
#third column is P(Y==1)
GBM_valpredictions <- as.data.frame(GBM_valpred$pred)[,3]
labels <- as.data.frame(val_hex[,y])[,1]

#AUC expected
cvAUC::AUC(predictions = GBM_valpredictions, labels = labels)
###############
###########################################################################
# Check how each learner did so you can further tune 
##########################################################################

L <- length(GBM_learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(GBM_valpred$basepred)[,l], labels = labels)) 
data.frame(GBM_learner, auc)
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
submitter(hutest$mbr_id,GBM_predictions,"husubmission_model5") #Public leadboard score is 0.
###########################################################################################################
#Model 6
GBM_learner <- c(
  #  "h2o.gbm.1"
  "h2o.gbm.2"
  ,"h2o.gbm.3"
  ,"h2o.gbm.4"
  ,"h2o.gbm.5"
  ,"h2o.gbm.6"
  # ,"h2o.gbm.7"
  # ,"h2o.gbm.8"
  #,"h2o.gbm.24"
  #  ,"h2o.gbm.25"
  #  ,"h2o.gbm.26"
  #  ,"h2o.gbm.28"
  #  ,"h2o.gbm.34"
  #  ,"h2o.gbm.35"
  #  ,"h2o.gbm.36"
)

metalearner <- "h2o.glm.wrapper"

GBM_customfit <- h2o.ensemble(x = x12, y = y,
                              training_frame = train_hex,
                              family='binomial',
                              learner = GBM_learner,
                              metalearner =metalearner,
                              cvControl = list(V = 5))

#######################
#Check on Validation datasets
GBM_valpred <- predict(GBM_customfit, val_hex)
#third column is P(Y==1)
GBM_valpredictions <- as.data.frame(GBM_valpred$pred)[,3]
labels <- as.data.frame(val_hex[,y])[,1]

#AUC expected
cvAUC::AUC(predictions = GBM_valpredictions, labels = labels)
###############
###########################################################################
# Check how each learner did so you can further tune 
##########################################################################

L <- length(GBM_learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(GBM_valpred$basepred)[,l], labels = labels)) 
data.frame(GBM_learner, auc)
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
submitter(hutest$mbr_id,GBM_predictions,"husubmission_model6") #Public leadboard score is 0.
#################
#Model 7
x9= c("pri_cst","ethnct_ds_tx","dx_prspct_med_risk_qt","rx_prspct_ttl_risk_qt","mcare_prspct_med_risk_qt","loh_prspct_qt",
      "rx_inpat_prspct_ttl_risk_qt","cops2_qt","mcare_prspct_med_risk_qt_p")

GBM_learner <- c(
  "h2o.gbm.1"
  ,"h2o.gbm.2"
  ,"h2o.gbm.3"
  ,"h2o.gbm.4"
  ,"h2o.gbm.5"
  ,"h2o.gbm.6"
  ,"h2o.gbm.7"
  ,"h2o.gbm.8"
  #,"h2o.gbm.24"
  #  ,"h2o.gbm.25"
  #  ,"h2o.gbm.26"
  #  ,"h2o.gbm.28"
  #  ,"h2o.gbm.34"
  #  ,"h2o.gbm.35"
  #  ,"h2o.gbm.36"
)

metalearner <- "h2o.glm.wrapper"

GBM_customfit <- h2o.ensemble(x = x9, y = y,
                              training_frame = train_hex,
                              family='binomial',
                              learner = GBM_learner,
                              metalearner =metalearner,
                              cvControl = list(V = 5))

#######################
#Check on Validation datasets
GBM_valpred <- predict(GBM_customfit, val_hex)
#third column is P(Y==1)
GBM_valpredictions <- as.data.frame(GBM_valpred$pred)[,3]
labels <- as.data.frame(val_hex[,y])[,1]

#AUC expected
cvAUC::AUC(predictions = GBM_valpredictions, labels = labels)
###############
###########################################################################
# Check how each learner did so you can further tune 
##########################################################################

L <- length(GBM_learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(GBM_valpred$basepred)[,l], labels = labels)) 
data.frame(GBM_learner, auc)
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
submitter(hutest$mbr_id,GBM_predictions,"husubmission_model7") #Public leadboard score is 0.
###########################################################################################################
#Model 8
GBM_learner <- c(
  #  "h2o.gbm.1"
  "h2o.gbm.2"
  ,"h2o.gbm.3"
  ,"h2o.gbm.4"
  ,"h2o.gbm.5"
  ,"h2o.gbm.6"
  # ,"h2o.gbm.7"
  # ,"h2o.gbm.8"
  #,"h2o.gbm.24"
  #  ,"h2o.gbm.25"
  #  ,"h2o.gbm.26"
  #  ,"h2o.gbm.28"
  #  ,"h2o.gbm.34"
  #  ,"h2o.gbm.35"
  #  ,"h2o.gbm.36"
)

metalearner <- "h2o.glm.wrapper"

GBM_customfit <- h2o.ensemble(x = x9, y = y,
                              training_frame = train_hex,
                              family='binomial',
                              learner = GBM_learner,
                              metalearner =metalearner,
                              cvControl = list(V = 5))

#######################
#Check on Validation datasets
GBM_valpred <- predict(GBM_customfit, val_hex)
#third column is P(Y==1)
GBM_valpredictions <- as.data.frame(GBM_valpred$pred)[,3]
labels <- as.data.frame(val_hex[,y])[,1]

#AUC expected
cvAUC::AUC(predictions = GBM_valpredictions, labels = labels)
###############
###########################################################################
# Check how each learner did so you can further tune 
##########################################################################

L <- length(GBM_learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(GBM_valpred$basepred)[,l], labels = labels)) 
data.frame(GBM_learner, auc)
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
submitter(hutest$mbr_id,GBM_predictions,"husubmission_model8") #Public leadboard score is 0.	