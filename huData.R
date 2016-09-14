# KP 1st Data Challenage
# InitialDate: 5/27/2016 
data<-load("/hu/input/hu.RData")
str(hutest)
summary(hutrain)
#Problems needed to be taken care:
#1. Data type transformation for mbr_id (int to char), cms_hcc_1_ct - 177_ct (int to factor)
#Brainstorms:Problems may be needed to consider
# Age, age=0, mother-infant relationship
# other family as an entity 
# low-income Ppl: cms_mcare_advntg_in_cd,
# low-income family
head(hutrain)
View(hutrain)
ls.str(hutrain)
ls.str(hutest)
