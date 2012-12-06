\name{brnn}
\alias{brnn}
\title{brnn}
\description{
  The brnn function
  fits a two layer neural network as described in MacKay (1992) and Foresee and Hagan (1997). It uses the 
  Nguyen and Widrow algorithm (1990) to assign initial weights and the Gauss-Newton algorithm to 
  perform the optimization. This function implements the functionality of the function trainbr in Matlab 2010b.
}
\usage{
  brnn(y,X,neurons,epochs=1000,mu=0.005,mu_dec=0.1, 
                   mu_inc=10,mu_max=1e10,min_grad=1e-10,change = 0.001,cores=1,
                   verbose=TRUE,
                   Monte_Carlo = FALSE,tol = 1e-06, samples = 40)
}
\arguments{
  \item{y}{(numeric, \eqn{n}{n}) the response data-vector (NAs not  allowed).}
  \item{X}{(numeric, \eqn{n \times p}{n x p}) incidence matrix.}
  \item{neurons}{positive integer that indicates the number of neurons.}
  \item{epochs}{positive integer, maximum number of epochs(iterations) to train, default 1000.}
  \item{mu}{positive number that controls the behaviour of the Gauss-Newton optimization algorithm, default value 0.005.}
  \item{mu_dec}{positive number, is the mu decrease ratio, default value 0.1.}
  \item{mu_inc}{positive number, is the mu increase ratio, default value 10.}
  \item{mu_max}{maximum mu before training is stopped, strict positive number, default value \eqn{1\times 10^{10}}{1e10}.}
  \item{min_grad}{minimum gradient.}
  \item{change}{The program  will stop if the maximum (in absolute value) of the differences of the F function in 3 consecutive iterations is less than this quantity.}
  \item{cores}{Number of cpu cores to use for calculations (only available in UNIX-like operating systems).}
  \item{verbose}{logical, if TRUE will print iteration history.}
  \item{Monte_Carlo}{If TRUE it will estimate the trace of the inverse of the hessian using Monte Carlo procedures, see Bai et al. (1996) for 
                     more details. This routine calls the function estimate.trace() to perform the computations.}
  \item{tol}{numeric tolerance, a tiny number useful for checking convergenge in the Bai's algorithm. }
  \item{samples}{positive integer, number of Monte Carlo replicates to estimate the trace of the inverse, see Bai et al. (1996) for more details.}
}

\details{
  The software fits a two layer network as described in MacKay (1992) and Foresee and Hagan (1997). 
  The model is given by:
  
  \eqn{y_i=g(\boldsymbol{x}_i)+e_i = \sum_{k=1}^s w_k g_k (b_k + \sum_{j=1}^p x_{ij} \beta_j^{[k]}) + e_i, i=1,...,n}
  
  where:
  
  \itemize{
    \item{\eqn{e_i \sim N(0,\sigma_e^2)}{e_i ~ N(0,\sigma_e^2)}.}
    \item{\eqn{s} is the number of neurons.}
    \item{\eqn{w_k}{w_k} is the weight of the \eqn{k}{k}-th neuron, \eqn{k=1,...,s}{k=1,...,s}.}
    \item{\eqn{b_k}{b_k} is a bias for the \eqn{k}{k}-th neuron, \eqn{k=1,...,s}{k=1,...,s}.}
    \item{\eqn{\beta_j^{[k]}}{beta_j^{[s]}} is the weight of the \eqn{j}{j}-th input to the net, \eqn{j=1,...,p}{j=1,...,p}.}
    \item{\eqn{g_k(\cdot)}{g_k(.)} is the activation function, in this implementation \eqn{g_k(x)=\frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}}{g_k(x)=(exp(x)-exp(-x))/(exp(x)+exp(-x))}.}
   }  

   The software will minimize 
 
   \deqn{F=\beta E_D + \alpha E_W}{F=beta*E_D +  alpha *E_W}
 
   where 
   \itemize{
     \item{\eqn{E_D=\sum_{i=1}^n (y_i-\hat y_i)^2}{y_i-\hat y_i}, i.e. the error sum of squares.}
     \item{\eqn{E_W}{E_W} is the sum of squares of network parameters (weights and biases).}
     \item{\eqn{\beta=\frac{1}{2\sigma^2_e}}{beta=1/(2*sigma^2_e)}.}
     \item{\eqn{\alpha=\frac{1}{2\sigma_\theta^2}}{alpha=1/(2*sigma_theta^2)}, \eqn{\sigma_\theta^2}{sigma_theta^2} is a dispersion parameter for weights and biases.}
   }
  
}
\value{
A list containing: 
\item{$theta}{A list containing weights and biases. The first \eqn{s}{s} components of the list contains vectors with the estimated parameters for
              the \eqn{k}{k}-th neuron, i.e. \eqn{(w_k, b_k, \beta_1^{[k]},...,\beta_p^{[k]})'}.}
\item{$message}{String that indicates the stopping criteria for the training process.}
\item{$alpha}{alpha parameter.}
\item{$beta}{beta parameter.}
\item{$gamma}{effective number of parameters.}
\item{$Ew}{The sum of the squares of the bias and weights.}
\item{$Ed}{The sum of the squares between observed and predicted values.}
}

\references{

Bai, Z. J., M. Fahey and G. Golub (1996). "Some large-scale matrix computation problems." 
\emph{Journal of Computational and Applied Mathematics}  74(1-2): 71-89.

Foresee, F. D., and M. T. Hagan. 1997. "Gauss-Newton approximation to Bayesian regularization", 
\emph{Proceedings of the 1997 International Joint Conference on Neural Networks}.

Gianola, D. Okut, H., Weigel, K. and Rosa, G. 2011. "Predicting complex quantitative traits with Bayesian neural networks:
a case study with Jersey cows and wheat". \emph{BMC Genetics}.

MacKay, D. J. C. 1992. "Bayesian interpolation", \emph{Neural Computation}, vol. 4, no. 3, pp. 415-447.

Nguyen, D. and Widrow, B. 1990. "Improving the learning speed of 2-layer neural networks by choosing initial values of the adaptive weights",
\emph{Proceedings of the IJCNN}, vol. 3, pp. 21-26.

Paciorek, C. J. and Schervish, M. J. (2004). "Nonstationary Covariance Functions for
Gaussian Process Regression". In Thrun, S., Saul, L., and Scholkopf, B., editors, \emph{Advances
in Neural Information Processing Systems 16}. MIT Press, Cambridge, MA.

}

\examples{

\dontrun{

#Load the library
library(brnn)

###############################################################
#Example 1 
#Noise triangle wave function, similar to example 1 in Foresee and Hagan (1997)

cores=2

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

neurons=2
out=brnn(y,X,neurons=neurons,cores=cores)
cat("Message: ",out$reason,"\n")

#We fitted a 1-2-1 Bayesian regularized neural net by using the 
#trainbr function in Matlab 2010b and the results for the weights and biases are as follows. 
theta_true=list()
theta_true[[1]]=c(11.11,-0.61,1.08)
theta_true[[2]]=c(6.0942,1.46,-2.85)
theta_true[[3]]=0.63

yhat_Matlab=theta_true[[3]]+predictions.nn(X,theta_true,2)

#Compare the results obtained with the brnn function in R and trainbr in Matlab,
plot(x,y,xlim=c(0,1),ylim=c(-1.5,1.5),main="Bayesian Regularization for ANN 1-2-1")
lines(x,yhat_Matlab,col="red")
yhat_R=predictions.nn(X,out$theta,neurons)
lines(x,yhat_R,col="blue",lty=2)
legend("topright",legend=c("Matlab","R"),col=c("red","blue"),lty=c(1,2),bty="n")
 
###############################################################
#Example 2
#sin wave function, example in the Matlab 2010b demo.

cores=2

x = seq(-1,0.5,length.out=100)
y = sin(2*pi*x)+rnorm(length(x),sd=0.1)
X=as.matrix(x)
neurons=3
out=brnn(y,X,neurons=neurons,cores=cores)
cat("Message: ",out$reason,"\n")
plot(x,y)
yhat_R=predictions.nn(X,out$theta,neurons)
lines(x,yhat_R,col="blue",lty=2)


#################################################################
#Example 3

#2 Inputs and 1 output
#the data used in Paciorek and
#Schervish (2004). The data is from a two input one output function with Gaussian noise
#with mean zero and standard deviation 0.25.

cores=2

data(twoinput)
X=normalize(as.matrix(twoinput[,1:2]))
y=as.vector(twoinput[,3])

neurons=10
out=brnn(y,X,neurons=neurons,cores=cores)
cat("Message: ",out$reason,"\n")
   
f=function(x1,x2,theta,neurons) predictions.nn(X=cbind(x1,x2),theta,neurons)
x1=seq(min(X[,1]),max(X[,1]),length.out=50)
x2=seq(min(X[,1]),max(X[,1]),length.out=50)
z=outer(x1,x2,f,theta=out$theta,neurons=neurons) # calculating the density values
   
transformation_matrix=persp(x1, x2, z, 
                            main="Fitted model", 
                            sub=expression(y==italic(g)~(bold(x))+e),
                            col="lightgreen",theta=30, phi=20,r=50, d=0.1,expand=0.5,ltheta=90, lphi=180, 
                            shade=0.75, ticktype="detailed",nticks=5) 
points(trans3d(X[,1],X[,2], f(X[,1],X[,2],
       theta=out$theta,neurons=neurons), transformation_matrix), col = "red")

###############################################################
#Example 4
#Gianola et al. (2011).
#Warning, it will take a while, substitute the FALSE
#statement with the TRUE statement if you really want to run the example

cores=4
  
#Load the Jersey dataset
data(Jersey)
  
#Normalize inputs
y=normalize(pheno$yield_devMilk)
X=normalize(G)

#Fit the model with the FULL DATA
out=brnn(y=y,X=X,neurons=2,cores=cores)
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
out=brnn(y=ytraining,X=Xtraining,neurons=4)
cat("Message: ",out$reason,"\n")

#Plot the results
#Predicted vs observed values for the training set
par(mfrow=c(2,1))
yhat_R_training=predictions.nn(Xtraining,out$theta,neurons=4)
plot(ytraining,yhat_R_training,xlab=expression(hat(y)),ylab="y")
cor(ytraining,yhat_R_training)
  
#Predicted vs observed values for the testing set
yhat_R_testing=predictions.nn(Xtesting,out$theta,neurons=4)
plot(ytesting,yhat_R_testing,xlab=expression(hat(y)),ylab="y")
cor(ytesting,yhat_R_testing)

}

}

\keyword{models}
