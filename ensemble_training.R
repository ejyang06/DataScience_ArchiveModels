
#6/16: Idea: Training model on traning dataset and choose the best methods

submitter <- function(id,predictions,filename)
{ 
  submission<-cbind(id,predictions)
  colnames(submission) <- c("mbr_id", "prediction")
  submission <- as.data.frame(submission)
  #add your nuid by replacing p624626 
  filename = paste0("/hu/output/l103295",filename,"l103295.csv")
  write.csv(submission, filename,row.names = FALSE)
}
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