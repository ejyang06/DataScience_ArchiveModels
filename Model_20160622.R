### 6/22: Trying more parameters (44 para from GBM varImportant analysis)
rm(list = ls())


# Load into h2o
attach("/hu/input/hu.RData")
hutrain_hex = as.h2o(hutrain)
hutest_hex = as.h2o(hutest)
y <- "hu_01"
x <- hutrain_hex[,-c(141:145)]
x1 <- setdiff(names(x), y)
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


x44=c(
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
  "cms_hcc_27_ct",
  "ethnct_ds_tx",
  "rx_inpat_prspct_ttl_risk_qt_p",
  "mcare_cncur_med_risk_qt_p",
  "cms_hcc_72_ct",
  "cops2_qt_p",
  "rx_cncur_med_risk_qt",
  "cms_hcc_25_ct",
  "age_yr_nb",
  "mcare_cncur_med_risk_qt",
  "er_stay_ct_pri",
  "paneled",
  "admt_ct_pri",
  "rx_elig_mm_ct",
  "cms_hcc_26_ct",
  "rx_inpat_prspct_ttl_risk_qt",
  "hedis_rdmt_indx_ct_pri",
  "dx_cncur_med_risk_qt",
  "cg_2012",
  "ae_elig_mm_ct",
  "cms_hcc_8_ct",
  "cms_hcc_7_ct",
  "dx_dem_prspct_med_risk_qt",
  "cms_hcc_81_ct",
  "cms_hcc_9_ct",
  "new_enrlee_score_qt",
  "cms_hcc_44_ct",
  "cms_hcc_174_ct",
  "cms_hcc_2_ct"
  
)

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
metalearner <- "h2o.glm.wrapper"

GBM_customfit <- h2o.ensemble(x = x44, y = y,
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
cvAUC::AUC(predictions = GBM_valpredictions, labels = labels) #0.9060534
###############
###########################################################################
# Check how each learner did so you can further tune 
##########################################################################

L <- length(GBM_learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(GBM_valpred$basepred)[,l], labels = labels)) 
data.frame(GBM_learner, auc)

#GBM_learner       auc
#1 h2o.gbm.wrapper 0.8986687
#2       h2o.gbm.1 0.8996869
#3       h2o.gbm.2 0.9003159
#4       h2o.gbm.3 0.9035217
#5       h2o.gbm.4 0.9049068
#6       h2o.gbm.5 0.9045194
#7       h2o.gbm.6 0.9058348
#8       h2o.gbm.7 0.9008913
#9       h2o.gbm.8 0.901771

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
submitter(hutest$mbr_id,GBM_predictions,"husubmission-ensembleGBM-44para") #Public leadboard score is 0.89X
############################################################################################################
