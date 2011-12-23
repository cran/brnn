rm(list=ls())
library(brnn)

#Example 3
#2 Inputs and 1 output
#the data used in Paciorek and
#Schervish (2004). The data is from a two input one output function with Gaussian noise
#with mean zero and standard deviation 0.25

data(twoinput)

X=normalize(as.matrix(twoinput[,1:2]))
y=as.vector(twoinput[,3])

out=brnn(y,X,neurons=10,cores=2)
cat("Message: ",out$reason,"\n")
   
f=function(x1,x2,theta,neurons) predictions.nn(X=cbind(x1,x2),theta,neurons)
x1=seq(min(X[,1]),max(X[,1]),length.out=50)
x2=seq(min(X[,1]),max(X[,1]),length.out=50)
z=outer(x1,x2,f,theta=out$theta,neurons=10) # calculating the density values
   
transformation_matrix=persp(x1, x2, z, 
                            main="Fitted model", 
                            sub=expression(y==italic(g)~(bold(x))+e),
                            col="lightgreen",theta=30, phi=20,r=50, d=0.1,expand=0.5,ltheta=90, lphi=180, 
                            shade=0.75, ticktype="detailed",nticks=5) 
points(trans3d(X[,1],X[,2], f(X[,1],X[,2],theta=out$theta,neurons=10), transformation_matrix), col = "red") 

