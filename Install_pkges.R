###########################################
# Installing Packages for training
#########################################

##load packages 
packages <- c("randomForest", "rpart", "nnet","e1071","caTools","party","gbm","snowfall","effects","RODBC","RMySQL","Rserve","h2o")
packages <- lapply(packages, FUN = function(x) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x, repos='http://cran.us.r-project.org', lib='/usr/lib64/R/library')
    suppressPackageStartupMessages(library(x, character.only = TRUE, warn.conflicts = FALSE, quietly = TRUE))
  }
})
#TODO : "Rserve","RODBC"
#R CMD INSTALL Rserve_1.8-5.tar.gz


###########################################
# Check Packages Installed on your computer
#########################################

ipack <- as.data.frame(installed.packages()[,c(1,3:4)])
#rownames(ipack) <- NULL
ipack <- ipack[is.na(ipack$Priority),1:2,drop=FALSE]
print(ipack, row.names=FALSE)