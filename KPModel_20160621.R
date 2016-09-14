
#6/21: Try to improve my models, selecting the important variables
rm(list = ls())
library(h2o)
localH2O = h2o.init(ip = '172.16.14.233', port = 54321,strict_version_check= FALSE)
h2o.removeAll() 

submitter <- function(id,predictions,filename)
{ 
  submission<-cbind(id,predictions)
  colnames(submission) <- c("mbr_id", "prediction")
  submission <- as.data.frame(submission)
  #add your nuid 
  filename = paste0("/hu/output/l103295",filename,"l103295.csv")
  write.csv(submission, filename,row.names = FALSE)
}

# Load into h2o
attach("/hu/input/hu.RData")
hutrain_hex = as.h2o(hutrain)
hutest_hex = as.h2o(hutest)
y <- "hu_01"
x <- hutrain_hex[,-c(141:145)]
x1 <- setdiff(names(x), y)
###########################################################################################


