rm(list=ls())
library(brnn)

#Warning, it will take a whil
#Load the Jersey dataset

data(Jersey)
y=normalize(pheno$yield_devMilk)
X1=normalize(G)
X2=normalize(D)
  
#Predictive power of the model using the SECOND set for 10 fold CROSS-VALIDATION
index=partitions==2
X1training=X1[!index,]
ytraining=y[!index]
X1testing=X1[index,]
ytesting=y[index]
X2training=X2[!index,]
X2testing=X2[index,]

#Fit the model for the TESTING DATA for Additive + Dominant
out=brnn.extended(y=ytraining,X1=X1training,
                  X2=X2training,neurons1=2,neurons2=2,cores=4,epochs=2000)
cat("Message: ",out$reason,"\n")

#Plot the results
#Predicted vs observed values for the training set
par(mfrow=c(2,1))
yhat_R_training=out$c_a*predictions.nn(X1training,out$theta1,2)
		+out$c_d*predictions.nn(X2training,out$theta2,2)
plot(ytraining,yhat_R_training,xlab=expression(hat(y)),ylab="y")
cor(ytraining,yhat_R_training)
  
#Predicted vs observed values for the testing set
yhat_R_testing=out$c_a*predictions.nn(X1testing,out$theta1,2)
              +out$c_d*predictions.nn(X2testing,out$theta2,2)
plot(ytesting,yhat_R_testing,xlab=expression(hat(y)),ylab="y")
cor(ytesting,yhat_R_testing)

