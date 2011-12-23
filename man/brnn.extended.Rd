\name{brnn.extended}
\alias{brnn.extended}
\title{brnn.extended}
\description{
  The brnn.extended function
  fits a two layer neural network as described in MacKay (1992) and Foresee and Hagan (1997). It uses the 
  Nguyen and Widrow algorithm (1990) to assign initial weights and the Gauss-Newton algorithm to 
  perform the optimization. The hidden layer contains two groups of neurons 
  that allow us to assign different prior distributions for two groups of input variables.
  
}
\usage{
  brnn.extended(y,X1,X2,neurons1,neurons2,epochs=1000,mu=0.005,mu_dec=0.1, 
                   mu_inc=10,mu_max=1e10,min_grad=1e-10,change = 0.001,
                   cores=1,verbose =TRUE)
}
\arguments{
  \item{y}{(numeric, \eqn{n}{n}) the response data-vector (NAs not  allowed).}
  \item{X1}{(numeric, \eqn{n \times p}{n x p}) incidence matrix for variables in group 1.}
  \item{X2}{(numeric, \eqn{n \times q}{n x q}) incidence matrix for variables in group 2.}
  \item{neurons1}{positive integer that indicates the number of neurons for variables in group 1.}
  \item{neurons2}{positive integer that indicates the number of neurons for variables in group 2.}
  \item{epochs}{positive integer, maximum number of epochs to train, default 1000.}
  \item{mu}{positive number that controls the behaviour of the Gauss-Newton optimization algorithm, default value 0.005.}
  \item{mu_dec}{positive number, is the mu decrease ratio, default value 0.1.}
  \item{mu_inc}{positive number, is the mu increase ratio, default value 10.}
  \item{mu_max}{maximum mu before training is stopped, strict positive number, default value \eqn{1\times 10^{10}}{1e10}.}
  \item{min_grad}{minimum gradient.}
  \item{change}{The program  will stop if the maximum (in absolute value) of the differences of the F 
                function in 3 consecutive iterations is less than this quantity.}
  \item{cores}{Number of cpu cores to use for calculations.}
  \item{verbose}{logical, if TRUE will print iteration history.}
}

\details{
  The software fits a two layer network as described in MacKay (1992) and Foresee and Hagan (1997). 
  The model is given by:
  
  \eqn{y_i=c_a \sum_{k=1}^{s_1} w_k^{1} g_k (b_k^{1} + \sum_{j=1}^p x1_{ij} \beta_j^{1[k]}) +
           c_d \sum_{k=1}^{s_2} w_k^{2} g_k (b_k^{2} + \sum_{j=1}^q x2_{ij} \beta_j^{2[k]})\,\,e_i, i=1,...,n}
  
  \itemize{
    \item{\eqn{e_i \sim N(0,\sigma_e^2)}{e_i ~ N(0,\sigma_e^2)}.}
    \item{\eqn{g_k(\cdot)}{g_k(.)} is the activation function, in this implementation \eqn{g_k(x)=\frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}}{g_k(x)=(exp(x)-exp(-x))/(exp(x)+exp(-x))}.}
  }
  
  The software will minimize 
 
  \deqn{F=\beta E_D + \alpha \theta_1' \theta_1 +\delta \theta_2' \theta_2 + \alpha_c c_a^2 + \delta_c c_d^2 }
  
  where 
  \itemize{
     \item{\eqn{E_D=\sum_{i=1}^n (y_i-\hat y_i)^2}{y_i-\hat y_i}, i.e. the sum of squared errors.}
     \item{\eqn{\beta=\frac{1}{2\sigma^2_e}}{beta=1/(2*sigma^2_e)}.}
   }
}
\value{
A list containing: 
\item{$theta1}{A list containing weights and biases. The first \eqn{s_1}{s_1} components of the list contain vectors with 
               the estimated parameters for the \eqn{k}{k}-th neuron, i.e. \eqn{(w_k^1, b_k^1, \beta_1^{1[k]},...,\beta_p^{1[k]})'}. 
               \eqn{s_1}{s_1} corresponds to neurons1 in the argument list.}
\item{$theta2}{A list containing weights and biases. The first \eqn{s_2}{s_2} components of the list contains vectors with 
               the estimated parameters for the \eqn{k}{k}-th neuron, i.e. \eqn{(w_k^2, b_k^2, \beta_1^{2[k]},...,\beta_q^{2[k]})'}.
                \eqn{s_2}{s_2} corresponds to neurons2 in the argument list.}
\item{$c_a}{An estimate of \eqn{c_a}{c_a}}
\item{$c_d}{An estimate of \eqn{c_d}{c_d}}
\item{$message}{String that indicates the stopping criteria for the training process.}
}

\references{

Foresee, F. D., and M. T. Hagan. 1997. "Gauss-Newton approximation to Bayesian regularization", 
\emph{Proceedings of the 1997 International Joint Conference on Neural Networks}.

MacKay, D. J. C. 1992. "Bayesian interpolation", \emph{Neural Computation}, vol. 4, no. 3, pp. 415-447.

Nguyen, D. and Widrow, B. 1990. "Improving the learning speed of 2-layer neural networks by choosing initial values of the adaptive weights",
\emph{Proceedings of the IJCNN}, vol. 3, pp. 21-26.

}


\examples{

\dontrun{
#Load the libraries 

library(brnn)

###############################################################
#Example 5: Additive + Dominant

#Make cluster
cores=4
  
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
                  X2=X2training,neurons1=2,neurons2=2,epochs=2000,cores=cores)
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
  
}
 
}

\keyword{models}
