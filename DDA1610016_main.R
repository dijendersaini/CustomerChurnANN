#-------------------------------------------------------------------------------------------------------#
#                           Neural Network assignment - Dijender Saini DDA1610016
#-------------------------------------------------------------------------------------------------------#

# Load libraries
library("MASS")
library("car")
library("h2o")
library("caret")
library("ggplot2")

# Set working directory
setwd("E:/IIITB/Course4/AssignmentNN/")

# Load csv file into a data frame "
churn <- read.csv("telecom_nn_train.csv")

#-------------------------------------------------------------------------------------------------------#
#                                   Business and Data Understanding
#-------------------------------------------------------------------------------------------------------#

# Business objective: To predict customer churn based on customer attributes and services subscribed

 
# Data understanding : Training data set contains 4930 observations with data of various customers

# The demographic/customer behavioural attribues which describe the customers in the dataset are :
# 1. gender : Numeric and Binary  
# 2. SeniorCitizen : Numeric and Binary  
# 3. Partner : Numeric and Binary  
# 4. Dependents : Numeric and Binary  
# 5. tenure : Numeric and continous
# 6. PhoneService : Numeric and Binary  
# 7. PaperlessBilling : Numeric and Binary  
# 8. MonthlyCharges : Numeric and continous
# 9. TotalCharges : Numeric and continous
# 10. Churn(Response variable) :String and Binary
# 11. MultipleLinesNo.phone.service : Numeric and Binary  
# 12. MultipleLinesYes : Numeric and Binary  
# 13. InternetServiceFiber.optic : Numeric and Binary  
# 14. InternetServiceNo : Numeric and Binary  
# 15. OnlineSecurityNo.internet.service : Numeric and Binary  
# 16. OnlineSecurityYes : Numeric and Binary  
# 17. OnlineBackupNo.internet.service : Numeric and Binary  
# 18. OnlineBackupYes : Numeric and Binary  
# 19. DeviceProtectionNo.internet.service : Numeric and Binary  
# 20. DeviceProtectionYes : Numeric and Binary  
# 21. TechSupportNo.internet.service : Numeric and Binary  
# 22. TechSupportYes : Numeric and Binary  
# 23. StreamingTVNo.internet.service : Numeric and Binary  
# 24. StreamingTVYes : Numeric and Binary  
# 25. StreamingMoviesNo.internet.service : Numeric and Binary  
# 26. StreamingMoviesYes : Numeric and Binary  
# 27. ContractOne.year : Numeric and Binary  
# 28. ContractTwo.year : Numeric and Binary  
# 29. PaymentMethodCredit.card..automatic. : Numeric and Binary  
# 30. PaymentMethodElectronic.check : Numeric and Binary  
# 31. PaymentMethodMailed.check : Numeric and Binary  

# Check for NA values
length(grep("TRUE",is.na(churn))) #6 NA values found
 
# Finding which column has NA values
list.NA<-""
for (i in c(1:ncol(churn)))
  {
 len<-length(grep("TRUE",is.na(churn[,i])))
 if(len > 0){
   list.NA<-paste(colnames(churn[i]),":",len,list.NA)
 }
}
list.NA # TotalCharges has 6 NA values

# Examining the records with NA values
NA.Records <- churn[which(is.na(churn$TotalCharges)),]

# Copying monthly charges to Total charges as tenure is zero
churn$TotalCharges<-ifelse(is.na(churn$TotalCharges),churn$MonthlyCharges,churn$TotalCharges)

# Outlier verification

# Continous numeric attributes are :
# 1. MonthlyCharges
# 2. TotalCharges
# 3. tenure

boxplot.stats(churn$MonthlyCharges) #No outlier
boxplot.stats(churn$TotalCharges) #No outlier
boxplot.stats(churn$tenure) #No outlier

response <-"Churn"
attributes <- setdiff(names(churn), response)

#-------------------------------------------------------------------------------------------------------#
#                                         EDA
#-------------------------------------------------------------------------------------------------------#

# More churn observed in customers with shorter tenure
plot1<-ggplot(churn,aes(churn$tenure,fill = Churn))
plot1+geom_histogram(binwidth=10,position = "dodge")+xlab("Tenure")+ggtitle("Tenure")

# More churn observed in customers with smaller annual amount
plot2<-ggplot(churn,aes(churn$TotalCharges,fill = Churn))
plot2+geom_histogram(binwidth=500,position = "dodge")+xlab("Total Charges")+ggtitle("Total Charges")

# Churn observed equally across genders
plot3<-ggplot(churn,aes(as.factor(churn$gender),fill = Churn))
plot3+geom_bar(position = "dodge")+xlab("Gender")+ggtitle("Gender")

# In percentage terms, Senior Citizens tend to churn more
plot4<-ggplot(churn,aes(as.factor(churn$SeniorCitizen),fill = Churn))
plot4+geom_bar(position = "dodge")+xlab("Senior Citizen")+ggtitle("Senior Citizen")

# Customers with no or max additional services churn less
plot5<-ggplot(churn,aes(as.factor(churn$StreamingTVYes+
                                    churn$TechSupportYes+
                                    churn$StreamingMoviesYes+
                                    churn$DeviceProtectionYes+
                                    churn$OnlineBackupYes+
                                    churn$OnlineSecurityYes+
                                    churn$MultipleLinesYes),fill = Churn))
plot5+geom_bar(position = "dodge")+xlab("No. of additional services")+ggtitle("Additional Services")


# Converting response variable to factor
churn[,response] <- as.factor(churn[,response])

# Creating training and validation data frames
# 30% of data in the training dataset is set as validation data

churn.rows<-nrow(churn)
validationsize <- churn.rows*0.3

subsample <- sample(nrow(churn),validationsize, replace=FALSE)

churn.validation.df<-churn[subsample,]
churn.train.df<-churn[setdiff(1:churn.rows,subsample),]

write.csv(churn.validation.df, file='telecom_churn_validation.csv', row.names=FALSE)
write.csv(churn.train.df, file='telecom_churn_train.csv', row.names=FALSE)

#-------------------------------------------------------------------------------------------------------#
#                                     Modelling using H2O
#-------------------------------------------------------------------------------------------------------#

# Initialize the h2o environment

h2o.init() 

churn.train.df.h2o <- h2o.importFile("telecom_churn_train.csv")
churn.validation.df.h2o  <- h2o.importFile("telecom_churn_validation.csv")
set.seed(1000)
#-------------------------------------------------------------------------------------------------------#
#                                         Without Epoch
#-------------------------------------------------------------------------------------------------------#
# With trials found the following :
  # - Increasing or decreasing the number of hidden layes from 6 decreases the accuracy
  # - Increasing or decreasing the number of neurons from 100 decreases the accuracy
  # - Activation function Tanh gives a better accuracy than Rectifier
NN.model0 <- h2o.deeplearning(x = attributes,
                              y = response,
                              training_frame = churn.train.df.h2o,
                              validation_frame = churn.validation.df.h2o,
                              distribution = "multinomial",
                              activation = "Tanh",
                              hidden = c(100,100,100,100,100,100),
                              l1 = 1e-5,
                              epochs = 0,
                              seed = 1000,
                              reproducible = T)
summary(NN.model0) 
#AUC:   Training: 0.7055955 
#       Validation: 0.7191753
h2o.mse(NN.model0) #0.2949216
h2o.rmse(NN.model0) #0.5430669
h2o.performance(NN.model0)
# Positive class : Churn=1
# Sensitivity(True positive rate) = 609/884 = 68.8%
# Specificity(True negative rate) = 1649/2567= 64.2%
# Accuracy = 65.4%


#-------------------------------------------------------------------------------------------------------#
#                                         With Epoch
#-------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#
#                                         Model 1
#-------------------------------------------------------------------------------------------------------#
# Aim of the modelling is to predict customers that will churn based on the customer attributes
# Telecom company can then focus on vulnerable customers and try to retain them
# Given this scenario, we would focus on predicting true class(Churn=Yes) as accurately as possible
# Focus will also be on to avoid overfitting and making model complex

# Starting with some random hyper paramters
NN.model1 <- h2o.deeplearning(x = attributes,
                             y = response,
                             training_frame = churn.train.df.h2o,
                             validation_frame = churn.validation.df.h2o,
                             distribution = "multinomial",
                             activation = "RectifierWithDropout",
                             hidden = c(200,200,200,200),
                             hidden_dropout_ratio = c(0.1, 0.1, 0.1, 0.1),
                             l1 = 1e-5,
                             seed=1000,
                             reproducible = T,
                             epochs = 500)
summary(NN.model1) 
#AUC: Training:0.8697716
#     Validation:0.8495773 
h2o.mse(NN.model1) #0.1242044
h2o.rmse(NN.model1) #0.3524264
h2o.performance(NN.model1)
# Positive class : Churn="Yes"
# Sensitivity(True positive rate) = 718/884 = 81.2%
# Specificity(True negative rate) = 1967/2567 = 76.6%
# Accuracy = 77.8%

#-------------------------------------------------------------------------------------------------------#
#                                         Model 2
#-------------------------------------------------------------------------------------------------------#
# Increasing number of layers increases the accuracy
NN.model2 <- h2o.deeplearning(x = attributes,
                              y = response,
                              training_frame = churn.train.df.h2o,
                              validation_frame = churn.validation.df.h2o,
                              distribution = "multinomial",
                              activation = "RectifierWithDropout",
                              hidden = c(200,200,200,200,200),
                              hidden_dropout_ratio = c(0.1, 0.1, 0.1, 0.1,0.1),
                              l1 = 1e-5,
                              seed=1000,
                              reproducible = T,
                              epochs = 1000)
summary(NN.model2) 
#AUC: Training: 0.8728413 
#     Validation: 0.8471881
h2o.mse(NN.model2) # 0.1236802
h2o.rmse(NN.model2) # 0.351682
h2o.performance(NN.model2)
# Positive class : Churn="Yes"
# Sensitivity(True positive rate) = 641/884 = 72.5%
# Specificity(True negative rate) = 2150/2567 = 83.7%
# Accuracy = 80.8%

#-------------------------------------------------------------------------------------------------------#
#                                         Model 3
#-------------------------------------------------------------------------------------------------------#
# Increasing number of layers further increases the accuracy but RMSE and MSE are also increasing
NN.model3 <- h2o.deeplearning(x = attributes,
                              y = response,
                              training_frame = churn.train.df.h2o,
                              validation_frame = churn.validation.df.h2o,
                              distribution = "multinomial",
                              activation = "RectifierWithDropout",
                              hidden = c(200,200,200,200,200,200),
                              hidden_dropout_ratio = c(0.1, 0.1, 0.1, 0.1,0.1, 0.1),
                              l1 = 1e-5,
                              seed=1000,
                              reproducible = T,
                              epochs = 1000)
summary(NN.model3) 
#AUC: Training: 0.8747358 
#     Validation: 0.8489058
h2o.mse(NN.model3) # 0.1220762
h2o.rmse(NN.model3) # 0.3493941
h2o.performance(NN.model3)
# Positive class : Churn="Yes"
# Sensitivity(True positive rate) = 742/884 = 83.9%
# Specificity(True negative rate) = 1949/2567 = 75.9%
# Accuracy = 77.9%

#-------------------------------------------------------------------------------------------------------#
#                                         Model 4
#-------------------------------------------------------------------------------------------------------#
# Increasing number of layers further to increases the accuracy 
# Apparently accuracy has increased a little but RMSE and MSE are also increasing
# Increasing the epoch doesn't have impact on the accuracy so epoch=1000 is found to be suitable
NN.model4 <- h2o.deeplearning(x = attributes,
                              y = response,
                              training_frame = churn.train.df.h2o,
                              validation_frame = churn.validation.df.h2o,
                              distribution = "multinomial",
                              activation = "RectifierWithDropout",
                              hidden = c(200,200,200,200,200,200,200),
                              hidden_dropout_ratio = c(0.1, 0.1, 0.1, 0.1,0.1, 0.1,0.1),
                              l1 = 1e-5,
                              seed=1000,
                              reproducible = T,
                              epochs = 1000)
summary(NN.model4) 
#AUC: Training: 0.870589 
#     Validation: 0.8469756
h2o.mse(NN.model4) # 0.1256348
h2o.rmse(NN.model4) # 0.3544499
h2o.performance(NN.model4)
# Positive class : Churn="Yes"
# Sensitivity(True positive rate) = 691/884 = 78.1%
# Specificity(True negative rate) = 2039/2567 = 79.4%
# Accuracy = 79.1%

#-------------------------------------------------------------------------------------------------------#
#                                         Model 5
#-------------------------------------------------------------------------------------------------------#
# On adding further hidden layes, accuracy of validation data starts falling. 
# RMSE has increased significantly
NN.model5 <- h2o.deeplearning(x = attributes,
                              y = response,
                              training_frame = churn.train.df.h2o,
                              validation_frame = churn.validation.df.h2o,
                              distribution = "multinomial",
                              activation = "RectifierWithDropout",
                              hidden = c(200,200,200,200,200,200,200,200),
                              hidden_dropout_ratio = c(0.1, 0.1, 0.1, 0.1,0.1, 0.1,0.1, 0.1),
                              l1 = 1e-5,
                              seed=1000,
                              reproducible = T,
                              epochs = 1000)
summary(NN.model5) 
#AUC: Training: 0.8664837 
#     Validation: 0.8490588
h2o.mse(NN.model5) # 0.1265758
h2o.rmse(NN.model5) # 0.3557749
h2o.performance(NN.model5)
# Positive class : Churn="Yes"
# Sensitivity(True positive rate) = 728/884 = 82.3%
# Specificity(True negative rate) = 1949/2567 = 75.9%
# Accuracy = 77.5%
#-------------------------------------------------------------------------------------------------------#
#                                         Final Model
#-------------------------------------------------------------------------------------------------------#
# Considerations while selecting the model
# 1. Sensitivity has to be high
# 2. Overall accuracy needs to be high
# 3. Model should not be very complex(# of layes, # of neurons etc.)

# Model NN.model3 is chosen as the final model which satisfies the above considerations

NN.model3 <- h2o.deeplearning(x = attributes,
                              y = response,
                              training_frame = churn.train.df.h2o,
                              validation_frame = churn.validation.df.h2o,
                              distribution = "multinomial",
                              activation = "RectifierWithDropout",
                              hidden = c(200,200,200,200,200,200),
                              hidden_dropout_ratio = c(0.1, 0.1, 0.1, 0.1,0.1, 0.1),
                              l1 = 1e-5,
                              seed=1000,
                              reproducible = T,
                              epochs = 1000)
summary(NN.model3) 
#AUC: Training: 0.8747358 
#     Validation: 0.8489058
h2o.mse(NN.model3) # 0.1220762
h2o.rmse(NN.model3) # 0.3493941
h2o.performance(NN.model3)
# Positive class : Churn="Yes"
# Sensitivity(True positive rate) = 742/884 = 83.9%
# Specificity(True negative rate) = 1949/2567 = 75.9%
# Accuracy = 77.9%
