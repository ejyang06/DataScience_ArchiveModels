###########################################################
# Sample code to Connect to mysql using R 
# Questions & concerns email:  Taposh 
# taposh.d.roy@kp.org
# Example : mysql
###########################################################


#Install RMySQL 
if (! ("RMySQL" %in% rownames(installed.packages()))) { install.packages("RMySQL") }

#load package
library(RMySQL)

# Connect to DB
mydb = dbConnect(MySQL(), user='ruser', password='ruser', dbname='tutorials', host='cskpcloudxp0356.cloud.kp.org')

# List Tables
dbListTables(mydb)

#List Fields
dbListFields(mydb,'tutorials_tbl')

#Query a table
rs = dbSendQuery(mydb, "select * from tutorials_tbl")

#See the data
data = fetch(rs, n=-1)