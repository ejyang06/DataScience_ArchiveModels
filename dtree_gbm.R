############################################################################################################
####                                                                                            ############
############################################################################################################
#######################
#Submitter function 
######################
submitter <- function(id,predictions,filename)
{ 
  submission<-cbind(id,predictions)
  colnames(submission) <- c("mbr_id", "prediction")
  submission <- as.data.frame(submission)
  #add your nuid by replacing p624626 
  filename = paste0("/hu/output/l103295",filename,"l103295.csv")
  write.csv(submission, filename,row.names = FALSE)
}


library(sqldf)


high<-sqldf('select * from hutrain where hu_01=1')
low<-sqldf('select * from hutrain where hu_01=0')
summary(high)
summary(low)

#comparing the total_encounter
summary(low$cnt_ttl_pri)
summary(high$cnt_ttl_pri)
histogram(low$cnt_ttl_pri)
histogram(high$cnt_ttl_pri)

#comparing pri_cst
summary(low$pri_cst)
summary(high$pri_cst)
histogram(low$pri_cst)
histogram(high$pri_cst)

# fitting classification trees
attach(hutrain)
summary(hu_01)
High1=ifelse(hu_01==1, "yes","no")
hutrain_var<-hutrain[,-c(141:145)]
hutrain1<-data.frame(hutrain_var,High1)
tree.hutrain1=tree(High1~.-ethnct_ds_tx - prmy_spkn_lang_cd -prmy_wrtn_lang_cd ,hutrain1)
ls.str(hutrain1)
summary(tree.hutrain1)
plot(tree.hutrain1)
text(tree.hutrain1,pretty=0)



##########################################################################
#Classification tree:
#  tree(formula = High1 ~ . - ethnct_ds_tx - prmy_spkn_lang_cd - 
#         prmy_wrtn_lang_cd, data = hutrain1)
#Variables actually used in tree construction:
#  [1] "dx_prspct_med_risk_qt"       "rx_inpat_prspct_ttl_risk_qt" "pri_cst"                    
#[4] "cms_hcc_130_ct"             
#Number of terminal nodes:  5 
#Residual mean deviance:  0.09599 = 49040 / 510900 
#Misclassification error rate: 0.01123 = 5736 / 510910 
###########################################################################
#  Pruning
###########################################################################
set.seed(214)
train.hu<-sample(1:nrow(hutrain1), nrow(hutrain1)/2)
test.hu<-hutrain1[-train.hu,]
High1.test=High1[-train.hu]
subtree.hutrain1=tree(High1~.-ethnct_ds_tx - prmy_spkn_lang_cd -prmy_wrtn_lang_cd ,hutrain1, subset=train.hu)
subhu.pred=predict(subtree.hutrain1,test.hu,type="class")
table(subhu.pred, High1.test)
(228+5032)/(494188+552)

set.seed(214)
cv.subtree=cv.tree(subtree.hutrain1, FUN=prune.misclass)
names(cv.subtree)
cv.subtree
par(mfrow=c(1,2))
plot(cv.subtree$sizecv.subtree$dev,type="b")
plot(cv.subtree$k,cv.subtree$dev,type="b")

# subtree5
prune.subtree5=prune.misclass(subtree.hutrain1,best=5)
plot(prune.subtree5)
text(prune.subtree5, pretty=0)
prune.pred5=predict(prune.subtree5,test.hu,type="class")
table(prune.pred5, High1.test)
(5032+228)/(494188+552)

# subtree4
prune.subtree4=prune.misclass(subtree.hutrain1,best=4)
plot(prune.subtree4)
text(prune.subtree4, pretty=0)
prune.pred4=predict(prune.subtree4,test.hu,type="class")
table(prune.pred4, High1.test)

##############################################################################
# Predicting the real hutest data
##############################################################################
hutest_prune.pred5=predict(prune.subtree5,hutest,type="class")
summary(hutest_prune.pred5)
head(hutest_prune.pred5)
submitter(hutest$mbr_id, hutest_prune.pred5,"husubmission-prune5")


########################################################################
# Running GBM with 4 paras
######################################################################
library(h2o)
localH2O = h2o.init(ip = '172.16.14.233', port = 54321,strict_version_check= FALSE)
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
View(impvariables)

#Select the variables important and do a GBM
x4 = c("pri_cst","dx_prspct_med_risk_qt","cms_hcc_130_ct","rx_inpat_prspct_ttl_risk_qt")

#Number of variables of x4
#4

#GBM Model with 8 parameters
hu.gbm2_4para = h2o.gbm(y = y, x = x4, training_frame = hutrain_hex, ntrees = 50, max_depth = 3, min_rows = 100)


# Model review
print(hu.gbm2_4para)


#Predict
gbm_h2o_predictions_4para = h2o.predict(hu.gbm2_4para, hutest_hex)

# Convert to R
gbm_prediction_4para = as.data.frame(gbm_h2o_predictions_4para$predict)
head(gbm_prediction_4para)
summary(gbm_prediction_4para)

#Creating a submission frame
submitter(hutest$mbr_id,gbm_prediction_4para$predict,"husubmission-gbm-4para")
###############################################################################################################
#Select the variables important and do a GBM
x3 = c("pri_cst","dx_prspct_med_risk_qt","cms_hcc_130_ct")

#Number of variables of x3
#3

#GBM Model with 8 parameters
hu.gbm2_3para = h2o.gbm(y = y, x = x3, training_frame = hutrain_hex, ntrees = 50, max_depth = 3, min_rows = 100)


# Model review
print(hu.gbm2_3para)


#Predict
gbm_h2o_predictions_3para = h2o.predict(hu.gbm2_3para, hutest_hex)

# Convert to R
gbm_prediction_3para = as.data.frame(gbm_h2o_predictions_3para$predict)
head(gbm_prediction_3para)
summary(gbm_prediction_3para)

#Creating a submission frame
submitter(hutest$mbr_id,gbm_prediction_3para$predict,"husubmission-gbm-3para")
