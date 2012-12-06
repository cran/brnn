rm(list=ls())
library(brnn)

##############################################################
#Example 4
#Gianola et al. (2011).
#Warning, it will take a while, substitute the FALSE
#statement with the TRUE statement if you really want to run the example
if(TRUE)
{
  #Load the Jersey dataset
  data(Jersey)
  
  #Normalize inputs
  y=normalize(pheno$yield_devMilk)
  X=normalize(G)

  #Fit the model with the FULL DATA
  out=brnn(y=y,X=X,neurons=2,cores=4)
  cat("Message: ",out$reason,"\n")

  #Obtain predictions
  yhat_R=predictions.nn(X,out$theta,neurons=2)
  plot(y,yhat_R)

  #Predictive power of the model using the SECOND set for 10 fold CROSS-VALIDATION
  index=partitions==2
  Xtraining=X[!index,]
  ytraining=y[!index]
  Xtesting=X[index,]
  ytesting=y[index]

  #Fit the model for the TESTING DATA
  out=brnn(y=ytraining,X=Xtraining,neurons=2,cores=4)
  cat("Message: ",out$reason,"\n")

  #Plot the results
  #Predicted vs observed values for the training set
  par(mfrow=c(2,1))
  yhat_R_training=predictions.nn(Xtraining,out$theta,neurons=2)
  plot(ytraining,yhat_R_training,xlab=expression(hat(y)),ylab="y")
  cor(ytraining,yhat_R_training)
  
  #Predicted vs observed values for the testing set
  yhat_R_testing=predictions.nn(Xtesting,out$theta,neurons=2)
  plot(ytesting,yhat_R_testing,xlab=expression(hat(y)),ylab="y")
  cor(ytesting,yhat_R_testing)
}
