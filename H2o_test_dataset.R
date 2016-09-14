##################################################################
# This is a demo of H2O's Capabilities as part of introductory class
# It imports a data set, parses it, and prints a summary
# Then, it runs RF with 50 trees, maximum depth of 100, using the iris class as the response
# Update Author : Taposh Roy
##################################################################

#H2O cloud is running so load the library
library(h2o)

#Connect to the ip address provided to your login.
h2o.init(ip="172.16.14.224", port=54323)

#You can load data from R data frame to h2o data frame
iris

#Move the sample Iris data-set to H2O data frame
iris.hex = as.h2o(iris)

#Look at Summary
summary(iris.hex)

#Data Munging - Overloaded R commands
dim(iris.hex)
nrow(iris.hex)
ncol(iris.hex) 
head(iris.hex) 
tail(iris.hex) 
colnames(iris.hex)

#Data Munging - H2O commands
h2o.mean(iris.hex)
#Median
h2o.median(iris.hex[1]) 
#Optional, the index of the column whose domain is to be returned.
h2o.levels(iris.hex,5)


#H2O Command h2o.cut
#Divides the range of the H2O data into intervals and codes the values 
#according to which interval they fall in. The leftmost interval corresponds
#to the level one, the next is level two, etc.
# e.g. Cut sepal length column into intervals determined by min/max/quantiles
sepal_len.cut = h2o.cut(iris.hex$Sepal.Length, c(4.2, 4.8, 5.8, 6, 8))
head(sepal_len.cut)
summary(sepal_len.cut)


#H2O Command h2o.impute
# Perform simple imputation on a single vector by filling missing values 
# with aggregates computed on the "na.rm'd" vector. Additionally, it's 
# possible to perform imputation based on groupings of columns from within data; 
# these columns can be passed by index or name to the by parameter. 
# If a factor column is supplied, then the method must be one "mode". 
# Anything else results in a full stop.
# http://rpackages.ianhowson.com/cran/h2o/man/h2o.impute.html

impute_frame <- as.h2o(iris, destination_frame="iris")
summary(impute_frame)
# For Species column randomly replace 50 values with NA 
impute_frame[sample(nrow(impute_frame),50),5] <- NA  
#Check the NA
summary(impute_frame)
# impute with a group by
impute_frame <- h2o.impute(impute_frame, "Species", "mode", by=c("Sepal.Length", "Sepal.Width"))
summary(impute_frame)




#H2O Command h2o.merge
#http://rpackages.ianhowson.com/cran/h2o/man/h2o.merge.html
fcolor <- data.frame(fruit = c('apple', 'orange', 'banana', 'lemon', 'strawberry', 'blueberry'),
                     color = c('red', 'orange', 'yellow', 'yellow', 'red', 'blue'))
citrus <- data.frame(fruit = c('apple', 'orange', 'banana', 'lemon', 'strawberry', 'watermelon'),
                     citrus = c(FALSE, TRUE, FALSE, TRUE, FALSE, FALSE))
l.hex <- as.h2o(fcolor)
summary(l.hex)
r.hex <- as.h2o(citrus)
summary(r.hex)
fruit_colors <- h2o.merge(l.hex, r.hex, all.x = TRUE)
summary(fruit_colors)



#Run Random Forest with 50 Trees
iris.rf = h2o.randomForest(y = 5, x = c(1,2,3,4), training_frame = iris.hex, ntrees = 50, max_depth = 100)
#Look at the model
print(iris.rf)
#download your model in java
h2o.download_pojo(iris.rf)


#Gradient Boosting Model
iris.gbm = h2o.gbm(x = 1:4, y = 5, training_frame = iris.hex)
#Look at the model
print(iris.gbm)
#download your model in java
h2o.download_pojo(iris.gbm)
