
#6/24
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