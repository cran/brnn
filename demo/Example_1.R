rm(list=ls())
library(brnn)

#Example 1 
#Noise triangle wave function, similar to example 1 in Foresee and Hagan (1997)

#Generating the data
x1=seq(0,0.23,length.out=25)
y1=4*x1+rnorm(25,sd=0.1)
x2=seq(0.25,0.75,length.out=50)
y2=2-4*x2+rnorm(50,sd=0.1)
x3=seq(0.77,1,length.out=25)
y3=4*x3-4+rnorm(25,sd=0.1)
x=c(x1,x2,x3)
y=c(y1,y2,y3)
X=as.matrix(x)

out=brnn(y,X,neurons=2,cores=2)

cat("Message: ",out$reason,"\n")

#We fitted a 1-2-1 Bayesian regularized neural net by using the 
#trainbr function in Matlab 2010b and the results for the weights and biases are as follows. 
theta_true=list()
theta_true[[1]]=c(11.11,-0.61,1.08)
theta_true[[2]]=c(6.0942,1.46,-2.85)
theta_true[[3]]=0.63

yhat_Matlab=theta_true[[3]]+predictions.nn(X,theta_true,neurons=2)

#Compare the results obtained with the brnn function in R and trainbr in Matlab,
plot(x,y,xlim=c(0,1),ylim=c(-1.5,1.5),
     main="Bayesian Regularization for ANN 1-2-1")
lines(x,yhat_Matlab,col="red")
yhat_R=predictions.nn(X,out$theta,neurons=2)
lines(x,yhat_R,col="blue",lty=2)
legend("topright",legend=c("Matlab","R"),col=c("red","blue"),lty=c(1,2),bty="n") 
