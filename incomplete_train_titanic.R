
incomplete.dt <- read.csv("~/C:\\Users\\L103295\\Desktop\\titanic_train_incomplete.csv")

#For complete dataset: Model will be ~age + pclass + sex
# Neural network
require(nnet)

mdl <- nnet(  as.factor(died) ~ sex + age + pclass, data=na.omit(subset(complete.dt, new_new_tt=="trn")), size=2)                      #Build neural net.
mdl
complete.dt$prd.ann <- predict(mdl, newdata=complete.dt)                                         #Get prediction.
rm(mdl)                                                                          #Clean up.



#incomplete
#SVM

require(e1071)

mdl <- svm(  as.factor(died) ~ sex + pclass
             , data=subset(incomplete.dt, new_new_tt=="trn"), probability=TRUE)                      #Build support vector machine.
incomplete.dt$prd.svm<- predict(mdl, newdata = incomplete.dt) #Populated prediction...carefully.
rm(mdl)


# Recursive partitioning.
require(randomForest)                                                            #Random forest.
require(party)                                                                   #Statistically based recursive partitioning.
require(gbm)                                                                     #Generalized boosted machines.

mdl <- ctree(  as.factor(died) ~ sex  + pclass
               , data=subset(incomplete.dt, new_new_tt=="trn"))                                      #Build conditional tree model..
plot(mdl)                                                                        #Fun plot.

incomplete.dt$prd.ctr <- unlist(predict(mdl, newdata=incomplete.dt, type="prob"))[1:(nrow(incomplete.dt)*2)%%2==0] #Get preciction...kind of painful.
rm(mdl)                                                                          #Clean up.

mdl <- randomForest(  as.factor(died) ~ sex + pclass 
                      , data=subset(incomplete.dt, new_tt=="trn"))                      #Build random forest model...works.
varImpPlot(mdl)                                                                  #Can produce some useful plots within the package.
incomplete.dt$prd.rf <- predict(mdl, newdata=incomplete.dt, type="prob")[,2]                         #Get prediction.
rm(mdl)                                                                          #Clean up.
#Build random forest model...fails because of missingness.
                  #Build random forest model...problem with factor not specified as a factor.
                     #Build random forest model...works.
                                                                  

mdl <- cforest(   as.factor(died) ~ sex + pclass
                  ,  data=subset(incomplete.dt, new_tt=="trn"))                                   #Build conditional tree model..
incomplete.dt$prd.cfst <- unlist(predict(mdl, newdata=incomplete.dt, type="prob"))[1:(nrow(incomplete.dt)*2)%%2==0] #Get preciction...kind of painful.
rm(mdl)                                                                          #Clean up.

mdl <- gbm(  died ~ sex +  pclass
             , data=subset(incomplete.dt, new_tt=="trn"), distribution="bernoulli", n.trees=10^4
             , interaction.depth=3)                                                #Build generalized boosted model (gradient boosted machine).
interact.gbm(x=mdl, data=subset(incomplete.dt, new_tt=="trn"), i.var=c("pclass","sex"))           

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

fncFH(mdl=mdl, data=subset(incomplete.dt, new_tt=="trn"))                                      #The function call.

bst <- gbm.perf(mdl,method="OOB")                                                #Show plot of performance and store best
incomplete.dt$prd.gbm <- predict(mdl, newdata=incomplete.dt, bst, type="response")                   #Get prediction.
rm(mdl, bst)                                                                     #Clean up.

# Neural net.
require(nnet)

mdl <- nnet(  as.factor(died) ~ sex  + pclass
              , data=subset(incomplete.dt, new_tt=="trn"), size=2)                      #Build neural net.
mdl
incomplete.dt$prd.ann <- predict(mdl, newdata=incomplete.dt)                                         #Get prediction.
rm(mdl)                                                                          #Clean up.

# Support vector machines.
#require(e1071)

#mdl <- svm(  as.factor(died) ~ sex  + pclass
             , data=subset(incomplete.dt, new_tt=="trn"), probability=TRUE)                      #Build support vector machine.
#incomplete.dt$prd.svm2[incomplete.dt$complete==1] <- new_ttr(predict(mdl, newdata=na.omit(incomplete.dt), probability=TRUE), "probabilities")[,2] #Populated prediction...carefully.
#rm(mdl)                                                                          #Clean up.

# Naive Bayes.
mdl <- naiveBayes(  as.factor(died) ~ sex + pclass
                    , data=subset(incomplete.dt, new_tt=="trn"))                                 #Build Naive Bayes model.
mdl
incomplete.dt$prd.nb <- predict(mdl, newdata=incomplete.dt[,names(mdl$tables)], type="raw")[,2]      #Get prediction...another funny one to avoid issues with non-represented variables appearing in the dataset fed to "predict".
rm(mdl)                                                                          #Clean up.

# Ensemble.

mdl <- ctree(  as.factor(died) ~ sex + pclass
               , data=subset(incomplete.dt, new_tt=="trn"))                                      #Build conditional tree model..
plot(mdl)                                                                        #Fun plot.
incomplete.dt$ctr.nod <- predict(mdl, newdata=incomplete.dt, type="node")                            #Get nodes for data.
rm(mdl)                                                                          #Clean up.
incomplete.dt$prd.ens <- NA                                                                #Allocate column for predictor.
nds <- sort(unique(incomplete.dt$ctr.nod))                                                 #Get unique list of nodes.
for(i in 1:length(nds)) {                                                        #Loop through nodes.
  mdl <- glm(  as.factor(died) ~ pclass + sex
               , data=subset(incomplete.dt, new_tt=="trn" & ctr.nod==nds[i]), family="binomial") #Build logistic regression model within each node group.
  incomplete.dt[incomplete.dt$ctr.nod==nds[i],]$prd.ens <- predict(mdl, newdata=incomplete.dt[incomplete.dt$ctr.nod==nds[i],], type="response") #Get node level predictions...only for rows in the particular node.
  rm(mdl)                                                                       #Clean up.
}
rm(nds,i)                                                                        #Clean up.

# Produce summary AUC table.
#require(caTools)                                                                 #Library that includes AUC calculations.

#colAUC(X=incomplete.dt$prd.gbm, y=incomplete.dt$died, plotROC = TRUE)                                                #Keep in mind that there are some missing values that get omminew_tted.
#smr <- data.frame( all=as.numeric(colAUC(X=incomplete.dt[,names(incomplete.dt)[grep("prd",names(incomplete.dt))]], y=incomplete.dt$died))
                   #,trn=as.numeric(colAUC(X=subset(incomplete.dt, new_tt=="trn")[,names(incomplete.dt)[grep("prd",names(incomplete.dt))]], y=subset(incomplete.dt, new_tt=="trn")$died))
                   #,tst=as.numeric(colAUC(X=subset(incomplete.dt, new_tt=="tst")[,names(incomplete.dt)[grep("prd",names(incomplete.dt))]], y=subset(incomplete.dt, new_tt=="tst")$died))
#)                                                               #Produce a dataframe with all results and split into testing and training.
#rownames(smr) <- names(incomplete.dt)[grep("prd",names(incomplete.dt))]                              #Change the rownames to make them more readable.
#smr$los <- with(smr, tst-trn)                                                    #Calculate loss from train to test.
#smr[order(smr$tst, decreasing=TRUE),]                                            #Print the results based on best testing performance.
#smr[order(smr$los, decreasing=TRUE),]                                            #Print the results based on least loss.


#write.csv(complete.dt, file="/tmp/complete_l103295.csv")
write.csv(incomplete.dt, file="/tmp/incomplete_l103295.csv")




# 