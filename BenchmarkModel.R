#Getting Started with R

#May 23rd - Starter code using GBM to get a Leaderboard Score of 0.8896938 using 24 parameters

########################################
# High Utilizer Starter Code  using GBM  
# Using H2o: LBScore 0.8896938                    
########################################

library(h2o)
attach("/hu/input/hu.RData")

#######################
#Submitter function 
######################
submitter <- function(id,predictions,filename)
{ 
  submission<-cbind(id,predictions)
  colnames(submission) <- c("mbr_id", "prediction")
  submission <- as.data.frame(submission)
  #add your nuid by replacing p624626 
  filename = paste0("/hu/output/",filename,"p624626.csv")
  write.csv(submission, filename,row.names = FALSE)
}

head(hutrain)
head(hutest)
nrow(hutrain)
nrow(hutest)

localH2O = h2o.init(ip = '172.16.14.240', port = 54321,strict_version_check= FALSE)
# Load into h2o
hutrain_hex = as.h2o(hutrain)
hutest_hex = as.h2o(hutest)

y <- "hu_01"
x <- hutrain_hex[,-c(141:145)]
x1 <- setdiff(names(x), y)


# Random Forest
hu.rf = h2o.randomForest(y = y, x = x1, training_frame = hutrain_hex)

#Lets look at variables since we will get penalized if used all variables
impvariables = h2o.varimp(hu.rf)

#Select the variables important and do a GBM
x2 = c("pri_cst","ethnct_ds_tx","dx_prspct_med_risk_qt","rx_prspct_ttl_risk_qt","mcare_prspct_med_risk_qt","loh_prspct_qt","cms_hcc_130_ct",
       "rx_inpat_prspct_ttl_risk_qt","cg_2014","cops2_qt","rx_prspct_ttl_risk_qt_p","mcare_prspct_med_risk_qt_p",
       "dx_cncur_med_risk_qt","age_yr_nb","ndi","race_ds_tx","loh_prspct_qt_p","rx_cncur_med_risk_qt",
       "hosp_day_ct_pri",   "rx_elig_mm_ct","cops_qt_p","dx_prspct_med_risk_qt_p","rx_inpat_prspct_ttl_risk_qt_p",
        "rx_cncur_med_risk_qt_p")

#Number of variables of x2
#24

#GBM
hu.gbm2 = h2o.gbm(y = y, x = x2, training_frame = hutrain_hex, ntrees = 50, max_depth = 3, min_rows = 100)


# Model review
print(hu.gbm2)

#Predict
h2o_predictions = h2o.predict(hu.gbm2, hutest_hex)

# Convert to R
prediction = as.data.frame(h2o_predictions$predict)
head(prediction)
summary(prediction)

#Creating a submission frame
submitter(hutest$mbr_id,prediction$predict,"husubmission-gbm-baselined-1")
Starter code to get a Leaderboard Score of 0.8068416 using 24 parameters

########################################
# High Utilizer Starter Code           #  
# Using H2o: LBScore 0.8068416                     
########################################

library(h2o)
attach("/hu/input/hu.RData")

#######################
#Submitter function 
######################
submitter <- function(id,predictions,filename)
{ 
  submission<-cbind(id,predictions)
  colnames(submission) <- c("mbr_id", "prediction")
  submission <- as.data.frame(submission)
  #add your nuid by replacing p624626 
  filename = paste0("/hu/output/",filename,"p624626.csv")
  write.csv(submission, filename,row.names = FALSE)
}

head(hutrain)
head(hutest)
nrow(hutrain)
nrow(hutest)

localH2O = h2o.init(ip = '172.16.14.240', port = 54321,strict_version_check= FALSE)
# Load into h2o
hutrain_hex = as.h2o(hutrain)
hutest_hex = as.h2o(hutest)

# Perform feature engineering
y <- "hu_01"
x <- hutrain_hex[,-c(141:145)]
x1 <- setdiff(names(x), y)

# Random Forest
hu.rf = h2o.randomForest(y = y, x = x1, training_frame = hutrain_hex)

#Lets look at variables since we will get penalized if used all variables
impvariables = h2o.varimp(hu.rf)

#Select the variables important and do a GBM
x2 = c("pri_cst","ethnct_ds_tx","dx_prspct_med_risk_qt","rx_prspct_ttl_risk_qt","mcare_prspct_med_risk_qt","loh_prspct_qt","cms_hcc_130_ct",
       "rx_inpat_prspct_ttl_risk_qt","cg_2014","cops2_qt","rx_prspct_ttl_risk_qt_p","mcare_prspct_med_risk_qt_p",
       "dx_cncur_med_risk_qt","age_yr_nb","ndi","race_ds_tx","loh_prspct_qt_p","rx_cncur_med_risk_qt",
       "hosp_day_ct_pri",   "rx_elig_mm_ct","cops_qt_p","dx_prspct_med_risk_qt_p","rx_inpat_prspct_ttl_risk_qt_p",
        "rx_cncur_med_risk_qt_p")

#Number of variables of x2
#24

# Real Random Forest limiting variables
hu.rf2 = h2o.randomForest(y = y, x = x2, training_frame = hutrain_hex,ntrees = 100,mtries=10, nbins=5,seed=756)

# Model review
print(hu.rf2)

#Predict
h2o_predictions = h2o.predict(hu.rf2, hutest_hex)

# Convert to R
prediction = as.data.frame(h2o_predictions$predict)
head(prediction)
summary(prediction)

#Creating a submission frame
submitter(hutest$mbr_id,prediction$predict,"husubmission-rf-baselined-1")
