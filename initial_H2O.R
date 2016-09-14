########################################
# Initiate h2o
#######################################
#install.packages("h2o")
library(h2o)
h2o.init()



#install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/master/3347/R")))


#require('devtools')

# The following two commands remove any previously installed H2O packages for R.
#if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
#if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

# Next, we download packages that H2O depends on.
#if (! ("methods" %in% rownames(installed.packages()))) { install.packages("methods") }
#if (! ("statmod" %in% rownames(installed.packages()))) { install.packages("statmod") }
#if (! ("stats" %in% rownames(installed.packages()))) { install.packages("stats") }
#if (! ("graphics" %in% rownames(installed.packages()))) { install.packages("graphics") }
#if (! ("RCurl" %in% rownames(installed.packages()))) { install.packages("RCurl") }
#if (! ("jsonlite" %in% rownames(installed.packages()))) { install.packages("jsonlite") }
#if (! ("tools" %in% rownames(installed.packages()))) { install.packages("tools") }
#if (! ("utils" %in% rownames(installed.packages()))) { install.packages("utils") }

# Now we download, install and initialize the H2O package for R.
#install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/master/3347/R")))

#Install libraries
library(dplyr)
library(ROCR)
library(pROC)
library(h2o)

#Instantiate H2o Server
#Note: Point to production IP to get production Data
# 172.16.14.226 is test IP
h2oserver <- h2o.init(ip='172.16.14.226')

#Data where the file is
pathToData <- "hdfs://nameservice1/group/lzetg/idr/episodeAttribution/"

#Import data.
#case 1: when # of categoricals are less than 9M
episode.hex <- h2o.importFile(path=pathToData,destination_frame = "episode.hex")

#Case : 2 whem # of categoriacals are more than 10M
episode.hex <- h2o.importFile(path=pathToData, parse=FALSE)
episode.hex = h2o.parseRaw(data=episode.hex,destination_frame = "episode.hex",col.types = c("string", rep(x = "numeric", times = 18)))

#Only use when you Import from Flow (http://172.16.14.226:54321/flow/index.html)
#episode.hex = h2o.getFrame('part_m_00000.hex')

summary(episode.hex)

#h2o.removeAll()
#h2o.ls()

#Garbage Collection 
h2o:::.h2o.gc()