rm(list=ls())
library(brnn)

#Example 2
#sin wave function, example in the Matlab 2010b demo.

x = seq(-1,0.5,length.out=100)
y = sin(2*pi*x)+rnorm(length(x),sd=0.1)
X=as.matrix(x)
out=brnn(y,X,neurons=3,cores=2)
cat("Message: ",out$reason,"\n")
plot(x,y)
yhat_R=predictions.nn(X,out$theta,neurons=3)
lines(x,yhat_R,col="blue",lty=2)

