###########################################
# Check Packages Installed on your computer
#########################################

ipack <- as.data.frame(installed.packages()[,c(1,3:4)])
#rownames(ipack) <- NULL
ipack <- ipack[is.na(ipack$Priority),1:2,drop=FALSE]
print(ipack, row.names=FALSE)