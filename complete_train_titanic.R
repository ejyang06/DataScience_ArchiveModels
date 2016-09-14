complete.dt <- read.csv("~/C:\\Users\\L103295\\Desktop\\titanic_train_complete.csv")# Recursive partitioning.
require(randomForest)                                                            #Random forest.
require(party)                                                                   #Statistically based recursive partitioning.
require(gbm)                                                                     #Generalized boosted machines.

mdl <- ctree(  as.factor(died) ~ sex  + pclass +age
               , data=subset(complete.dt, new_new_tt=="trn"))                                      #Build conditional tree model..
plot(mdl)                                                                        #Fun plot.

complete.dt$prd.ctr <- unlist(predict(mdl, newdata=complete.dt, type="prob"))[1:(nrow(complete.dt)*2)%%2==0] #Get preciction...kind of painful.
rm(mdl)                                                                          #Clean up.

mdl <- randomForest(  as.factor(died) ~ sex + pclass + age
                      , data=subset(complete.dt, new_tt=="trn"))                      #Build random forest model...works.
varImpPlot(mdl)                                                                  #Can produce some useful plots within the package.
complete.dt$prd.rf <- predict(mdl, newdata=complete.dt, type="prob")[,2]                         #Get prediction.
rm(mdl)                                                                          #Clean up.
#Build random forest model...fails because of missingness.
#Build random forest model...problem with factor not specified as a factor.
#Build random forest model...works.


mdl <- cforest(   as.factor(died) ~ sex + pclass
                  ,  data=subset(complete.dt, new_tt=="trn"))                                   #Build conditional tree model..
complete.dt$prd.cfst <- unlist(predict(mdl, newdata=complete.dt, type="prob"))[1:(nrow(complete.dt)*2)%%2==0] #Get preciction...kind of painful.
rm(mdl)                                                                          #Clean up.

mdl <- gbm(  died ~ sex +  pclass
             , data=subset(complete.dt, new_tt=="trn"), distribution="bernoulli", n.trees=10^4
             , interaction.depth=3)                                                #Build generalized boosted model (gradient boosted machine).
interact.gbm(x=mdl, data=subset(complete.dt, new_tt=="trn"), i.var=c("pclass","sex"))           

fncFH <- function(mdl, data) {                                                   #Create a function that returns the interactions strengths from the gbm object.
  itm <- c(mdl$var.names, rep("",max(0,mdl$interaction.depth-2)))               #List of variables to include--will incorporate 2-level interactions up to maximum level of interactions.
  tmp <- data.frame(t(combn(itm, m=mdl$interaction.depth)), stringsAsFactors=FALSE) #Get interaction combinations to try.
  names(tmp) <- tolower(names(tmp))                                             #Ensure variable names are lower case.
  tmp$fh <- NA                                                                  #Set Friedman H-statistic to NA.
  
  for(i in 1:nrow(tmp)) {                                                       #Loop through possible interactions.
    trm <- setdiff(as.character(tmp[i,]), c("","NA"))                          #Get list of terms to try.
    tmp[i,"fh"] <- interact.gbm(x=mdl, data=data, i.var=trm)                   #Produce measure of interaction.
  }
  
  return(tmp[order(tmp$fh, decreasing=TRUE),])                                  #Return dataframe by highest evidence of interaction.
}

fncFH(mdl=mdl, data=subset(complete.dt, new_tt=="trn"))                                      #The function call.

bst <- gbm.perf(mdl,method="OOB")                                                #Show plot of performance and store best
complete.dt$prd.gbm <- predict(mdl, newdata=complete.dt, bst, type="response")                   #Get prediction.
rm(mdl, bst)                                                                     #Clean up.

# Neural net.
require(nnet)

mdl <- nnet(  as.factor(died) ~ sex  + pclass
              , data=subset(complete.dt, new_tt=="trn"), size=2)                      #Build neural net.
mdl
complete.dt$prd.ann <- predict(mdl, newdata=complete.dt)                                         #Get prediction.
rm(mdl)                                                                          #Clean up.

# Support vector machines.
#require(e1071)

#mdl <- svm(  as.factor(died) ~ sex  + pclass
, data=subset(complete.dt, new_tt=="trn"), probability=TRUE)                      #Build support vector machine.
#complete.dt$prd.svm2[complete.dt$complete==1] <- new_ttr(predict(mdl, newdata=na.omit(complete.dt), probability=TRUE), "probabilities")[,2] #Populated prediction...carefully.
#rm(mdl)                                                                          #Clean up.

# Naive Bayes.
mdl <- naiveBayes(  as.factor(died) ~ sex + pclass
                    , data=subset(complete.dt, new_tt=="trn"))                                 #Build Naive Bayes model.
mdl
complete.dt$prd.nb <- predict(mdl, newdata=complete.dt[,names(mdl$tables)], type="raw")[,2]      #Get prediction...another funny one to avoid issues with non-represented variables appearing in the dataset fed to "predict".
rm(mdl)                                                                          #Clean up.

# Ensemble.

mdl <- ctree(  as.factor(died) ~ sex + pclass
               , data=subset(complete.dt, new_tt=="trn"))                                      #Build conditional tree model..
plot(mdl)                                                                        #Fun plot.
complete.dt$ctr.nod <- predict(mdl, newdata=complete.dt, type="node")                            #Get nodes for data.
rm(mdl)                                                                          #Clean up.
complete.dt$prd.ens <- NA                                                                #Allocate column for predictor.
nds <- sort(unique(complete.dt$ctr.nod))                                                 #Get unique list of nodes.
for(i in 1:length(nds)) {                                                        #Loop through nodes.
  mdl <- glm(  as.factor(died) ~ pclass + sex
               , data=subset(complete.dt, new_tt=="trn" & ctr.nod==nds[i]), family="binomial") #Build logistic regression model within each node group.
  complete.dt[complete.dt$ctr.nod==nds[i],]$prd.ens <- predict(mdl, newdata=complete.dt[complete.dt$ctr.nod==nds[i],], type="response") #Get node level predictions...only for rows in the particular node.
  rm(mdl)                                                                       #Clean up.
}
rm(nds,i)              