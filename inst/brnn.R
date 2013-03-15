# file brnn/brnn.R
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 or 3 of the License
#  (at your option).
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  A copy of the GNU General Public License is available at
#  http://www.r-project.org/Licenses/

#A package for Bayesian Regularized Neural Networks
#Author: Paulino Perez Rodriguez
#Madison, WI, Sep. 2012
#Birmingaham, Alabama, Jan. 2013

#WARNING: This is an experimental version

#Normalize a vector or matrix, the resulting all 
#resulting elements should be between -1 and 1
#z=2(x-base)/spread - 1
#where base=min(x), spread=max(x)-base

normalize=function(x,base,spread)
{
	if(is.matrix(x))
	{
		return(sweep(sweep(x,2,base,"-"),2,2/spread,"*")-1)
	}else{
	   	return(2*(x-base)/spread-1)
	}
}

#Go back to the original scale
#x=base+0.5*spread*(z+1)
un_normalize=function(x,base,spread)
{
	if(is.matrix(x))
	{
		return(sweep(sweep(x+1,2,0.5*spread,"*"),2,base,"+"))
	}else{
		return(base+0.5*spread*(x+1))		
	}	
}

ii=function(element=1,times=1)
{
  return(diag(rep(element,times)))
}

#the functions tanh and sech already defined in util.c, 
#make a wrapper here!!!!! 

#tanh function
tanh=function(x)
{
  (exp(2*x)-1)/(exp(2*x)+1)
}

#tansig function, the evaluation of this function 
#is faster than tanh and the derivative is the same!!!
tansig=function(x) 2/(1+exp(-2*x)) - 1 

#sech function
sech=function(x)
{
  2*exp(x)/(exp(2*x)+1)
}


#This function will obtain predicted values for a basic NN
#It uses an omp version of the routine predictions.nn to do the work
#This is an internal function
predictions.nn.C=function(vecX,n,p,theta,neurons,cores=1)
{
   yhat=rep(NA,n)
   vectheta=as.vector(unlist(theta))
   out=.Call("predictions_nn",as.double(vecX), as.integer(n), 
          as.integer(p), as.double(vectheta), 
          as.integer(neurons),
          as.double(yhat),
          as.integer(cores))
   yhat=out[[1]]
   return(yhat)
}

#This function will calculate the sum of the squares of the weights and biases
Ew=function(theta)
{
   sum((unlist(theta))^2)
}

jacobian=function(vecX,n,p,npar,theta,neurons,cores=1)
{
      vectheta=as.vector(unlist(theta))
      vecJ=rep(NA,n*npar)
      out=.Call("jacobian",as.double(vecX),as.integer(n),as.integer(p),
                as.double(vectheta),as.integer(neurons),
                as.double(vecJ),as.integer(cores))
      J=matrix(out[[1]],nrow=n,ncol=npar)
      return(J)
}

#Function to initialize the weights and biases in a neural network
#It uses the Nguyen-Widrow method
#FIXME:
#WARNING: It asumes that the inputs and outputs are between -1 and 1.
initnw=function(neurons,p,n,npar)
{
   theta=list()
   for(i in 1:neurons)
   {
        theta[[i]]=runif(npar/neurons,-0.5,0.5)
   }

   cat("Nguyen-Widrow method\n")

   if(p==1)
   {
       scaling_factor=0.7*neurons;
       cat("Scaling factor=",scaling_factor,"\n")
       for(i in 1:neurons)
       {
          lambda=theta[[i]]
          weight=lambda[1]
          bias=lambda[2]
          lambda=lambda[-c(1,2)]
          lambda=scaling_factor
          bias=runif(1,-scaling_factor,scaling_factor)
          theta[[i]]=c(weight,bias,lambda)
       }
   }else{
       scaling_factor=0.7*(neurons)^(1.0/n)
       cat("Scaling factor=",scaling_factor,"\n")
       b=seq(-1,1,length.out=neurons)
       for(i in 1:neurons)
       {
           lambda=theta[[i]]
           weight=lambda[1]
           bias=lambda[2]
           lambda=lambda[-c(1,2)]         
           norm=sqrt(sum(lambda^2))
           lambda=scaling_factor*lambda/norm
           bias=scaling_factor*b[i]*sign(lambda[1])
           theta[[i]]=c(weight,bias,lambda)
       }
   }   
   return(theta)
}

#Estimate the trace of the inverse of a positive definite and symmetric matrix
#Bai, Z. J., M. Fahey and G. Golub (1996). 
#"Some large-scale matrix computation problems." 
#Journal of Computational and Applied Mathematics 74(1-2): 71-89.
#A the matrix,
#tol tiny number, numeric tolerance
#B: Monte Carlo samples

estimate.trace=function(A,tol=1E-6,samples=40,cores=1)
{
    B=as.vector(A)
    n=nrow(A)
    m=1
    lm=1
    u=rep(0,n*(m+lm))
    d=rep(0,m+lm)
    ans=0
    rz=runif(n*samples)
    
    values=.C("extreme_eigenvalues",as.double(B),as.double(u),as.double(d),as.integer(n),
              as.integer(m),as.integer(lm),as.double(tol))[[3]]
    out=.Call("estimate_trace",as.double(B),as.integer(n),as.double(values[2]),as.double(values[1]),
           as.double(tol),as.integer(samples),as.integer(cores),as.double(rz),as.double(ans))
    ans=out[[1]]
    ans
}

#Function to do Bayesian Regularization for a network with a single layer and S neurons
#normalize, logical if TRUE will rescale inputs and output in the interval [-1,1]
#mu, strict pos scalar, default value 0.005
#mu_dec,strict positive scalar, is the mu decrease ratio, default value 0.1
#mu_inc, strict positive scalar, is the mu increase ratio, default value 10
#mu_max, maximum mu before training is stopped, strict pos scalar, default value 1e10
#min_grad 1e-10  minimum performance gradient
#change, if the augmented sum of squares is smaller than this number in 3 consecutive iterations the program will stop
#cores, number of threads to be used in calculations, only useful in UNIX like OS
#Monte_Carlo, if TRUE will estimate the trace of the Hessian using Monte Carlo, see estimate.trace for more details
#tol, tolerance, argument for estimate.trace
#samples number of Monte Carlo reps used for Monte Carlo estimates, argument for estimate.trace

brnn=function(y,X,neurons,normalize=TRUE,epochs=1000,mu=0.005,mu_dec=0.1, mu_inc=10,mu_max=1e10,
              min_grad=1e-10,change=0.001,cores=1,verbose=TRUE,
              Monte_Carlo=FALSE,tol=1E-6,samples=40)
{
   reason="UNKNOWN";

   #Checking that the imputs are ok
   if(!is.vector(y)) stop("y must be a vector\n")
   if(!is.matrix(X)) stop("X must be a matrix\n")
   
   if(normalize)
   {
        X_base=apply(X,2,min)
        X_spread=apply(X,2,max)-X_base 
        X_normalized=normalize(X,base=X_base,spread=X_spread)
   		
   		y_base=min(y)
   		y_spread=max(y) - y_base
   		y_normalized=normalize(y,base=y_base,spread=y_spread)
   		
   }else{
   		y_normalized=y
   		y_base=NULL
   		y_spread=NULL
   		
   		X_normalized=X
   		X_base=NULL
   		X_spread=NULL
   }
   
   vecX=as.vector(X_normalized)
   
   #Initializing parameters for the net
   p=ncol(X_normalized)
   n=length(y_normalized)
   
   #neurons is the number of neurons
   #the first 1 corresponds to the weight for the tansig function, i.e. weight*tansig(.)
   #the second 1 corresponds to the bias in tansig(.) function, .= bias + xi[1]*beta[1]+...+xi[p]*beta[p]
   #p corresponds to the number of covariates
   npar=neurons*(1+1+p)
   cat("Number of parameters (weights and biases) to estimate:",npar,"\n")

   theta=initnw(neurons,p,n,npar)
 
   gamma=npar
   
   e=y_normalized-predictions.nn.C(vecX,n,p,theta,neurons,cores)

   Ed=sum(e^2)
   beta=(n - gamma)/(2*Ed); if(beta<0) beta=1;
   Ew=Ew(theta)
   alpha=gamma/(2*Ew)

   epoch=1
   flag_gradient=TRUE
   flag_mu=TRUE
   flag_change_F=TRUE
   flag_change_Ed=TRUE

   F_history=numeric()

   C_new=0

   while(epoch<=epochs & flag_mu & flag_change_Ed & flag_change_F)
   {
      if(verbose)
      {
		cat("----------------------------------------------------------------------\n")
		cat("Epoch=",epoch,"\n")
      }
      
      J=jacobian(vecX,n,p,npar,theta,neurons,cores)
      
      H=crossprod(J)
      
      e=y_normalized-predictions.nn.C(vecX,n,p,theta,neurons,cores)

      #g=2*beta*t(J)%*%e+2*alpha*unlist(theta)
      g=2*as.vector((beta*t(e)%*%J+alpha*unlist(theta)))  #is it faster?
      mg=max(abs(g))
      flag_gradient=mg>min_grad
      Ed=sum(e^2)
      Ew=Ew(theta)
      C=beta*Ed+alpha*Ew
      if(verbose)
      {
		cat("C=",C,"\tEd=",Ed,"\tEw=",Ew,"\n")
		cat("gradient=",mg,"\n")
      }
       
      F_history[epoch]=C_new

      if(epoch>3)
      {
          if(max(abs(diff(F_history[(epoch-3):epoch])))<change) 
          {
              flag_change_F=FALSE;
              reason=paste("Changes in F= beta*SCE + alpha*Ew in last 3 iterations less than",change,sep=" ");
          }
      }
      flag_C=TRUE
      flag_mu=mu<=mu_max
      while(flag_C & flag_mu)
      {
	  	tmp=as.vector(unlist(theta)-solve(2*beta*H+ii(2*alpha+mu,npar),g))
	  	theta_new=list()
	  	for(i in 1:neurons)
	  	{
	      theta_new[[i]]=tmp[1:(2+p)]
	      tmp=tmp[-c(1:(2+p))]
	  	}
	  
      	e_new=y_normalized-predictions.nn.C(vecX,n,p,theta_new,neurons,cores)
	  	Ed=sum(e_new^2)
	  	Ew=Ew(theta_new)
	  	C_new=beta*Ed+alpha*Ew
	  	if(verbose) 
	  	{
	    	cat("C_new=",C_new,"\tEd=",Ed,"\tEw=",Ew,"\n")
	  	}
	  	if(C_new<C)
	  	{
	    	mu=mu*mu_dec
            if (mu < 1e-20) mu = 1e-20;
	    	flag_C=FALSE
	  	}else{
	    	mu=mu*mu_inc
	  	}
	  	if(verbose)
	  	{
	    	cat("mu=",mu,"\n")
	  	}
          flag_mu=mu<=mu_max
      }
      #Update all
      theta=theta_new
      epoch=epoch+1
      if(Monte_Carlo){
          gamma=npar-2*alpha*estimate.trace(2*beta*H+ii(2*alpha,npar),tol=tol,samples=samples,cores=cores) 
      }else{
          gamma=npar-2*alpha*sum(diag(solve(2*beta*H+ii(2*alpha,npar))))
      }
      alpha=gamma/(2*Ew)
      beta=(n-gamma)/(2*Ed)

      if(Ed<0.01)
      {
         flag_change_Ed=FALSE
         reason="SCE <= 0.01";
      }
      if(verbose)
      {
        cat("gamma=",round(gamma,4),"\t","alpha=",round(alpha,4),"\t","beta=",round(beta,4),"\n")
      }
   }
   if((epoch-1)==epochs) reason="Maximum number of epochs reached";
   if(!flag_mu) reason="Maximum mu reached";
   if(!flag_gradient) reason="Minimum gradient reached"; 
   if(!verbose)
   {
     cat("gamma=",round(gamma,4),"\t","alpha=",round(alpha,4),"\t","beta=",round(beta,4),"\n")
   }

   #answer
   out=list(theta=theta,alpha=alpha,beta=beta,gamma=gamma,Ed=Ed,Ew=Ew,F_history=F_history,reason=reason,epoch=epoch, 
            neurons=neurons,p=p,n=n,npar=npar,X_normalized=X_normalized,X_base=X_base,X_spread=X_spread,y_base=y_base,y_spread=y_spread,
            normalize=normalize)
   class(out)="brnn";
   
   #return the goodies
   return(out)
}

brnn.extended=function(y,X1,X2,neurons1,neurons2,epochs=1000,mu=0.005,mu_dec=0.1, mu_inc=10,mu_max=1e10,
                       min_grad=1e-10,change=0.001,cores=1,verbose=FALSE)
{
   reason="UNKNOWN";

   #Checking that the imputs are ok
   if(!is.vector(y)) stop("y must be a vector\n")
   if(!is.matrix(X1)) stop("X1 must be a matrix\n") 
   if(!is.matrix(X2)) stop("X2 must be a matrix\n")
   vecX1=as.vector(X1)
   vecX2=as.vector(X2)
   
   #Initializing parameters for the net
   p=ncol(X1)
   q=ncol(X2)
   n=length(y)
   
   #neurons is the number of neurons
   #the first 1 corresponds to the weight for the tansig function, i.e. weight*tansig(.)
   #the second 1 corresponds to the bias in tansig(.) function, .= bias + xi[1]*beta[1]+...+xi[p]*beta[p]
   #p corresponds to the number of covariates

   npar1=neurons1*(1+1+p)
   npar2=neurons2*(1+1+q)
   

   theta=initnw(neurons1+neurons2,p+q,n,npar=npar1+npar2)
   theta1=list()
   theta2=list()
   for(i in 1:neurons1)
   {
      theta1[[i]]=theta[[i]]
   }
   for(i in 1:neurons2)
   {
      theta2[[i]]=theta[[i+neurons1]]
   }
   
   alpha=0.01
   delta=0.01
   beta=1
   alpha_c=0.01
   delta_c=0.01
   w1=0.1
   w2=0.1
  
   epoch=1
   flag_gradient=TRUE
   flag_mu=TRUE
   flag_change_Ed=TRUE
   flag_change_F=TRUE

   F_history=numeric()
   C_new=0

   while(epoch<=epochs & flag_mu & flag_change_Ed & flag_change_F)
   {
      if(verbose)
      {
	cat("----------------------------------------------------------------------\n")
	cat("Epoch=",epoch,"\n")
      }
     
      #Parallel version  
      J1=jacobian(vecX1,n,p,npar1,theta1,neurons1,cores)
      J2=jacobian(vecX2,n,q,npar2,theta2,neurons2,cores)
      p1=predictions.nn.C(vecX1,n,p,theta1,neurons1,cores)
      p2=predictions.nn.C(vecX2,n,q,theta2,neurons2,cores)
      J=cbind(J1,J2,-p1,-p2)

    
      H=crossprod(J)

      e=y-w1*p1-w2*p2
      
      #g=2*beta*t(J)%*%e+2*c(alpha*unlist(theta1),delta*(unlist(theta2)),c(alpha_c*w1,delta_c*w2))
      g=2*as.vector((beta*t(e)%*%J+c(alpha*unlist(theta1),delta*(unlist(theta2)),c(alpha_c*w1,delta_c*w2))))  #is it faster?
      
      mg=max(abs(g))
      flag_gradient=mg>min_grad
      Ed=sum(e^2)
      E1=Ew(theta1)
      E2=Ew(theta2)
      E3=w1^2
      E4=w2^2
      C=beta*Ed+alpha*E1+delta*E2+alpha_c*E3+delta_c*E4
      if(verbose)
      {
	cat("C=",C,"\tEd=",Ed,"\tE1=",E1,"\tE2=",E2,"\tE3=",E3,"\tE4=",E4,"\n")
	cat("gradient=",mg,"\n")
      }

      F_history[epoch]=C_new
      
      if(epoch>4)
      {
          if(max(abs(diff(F_history[(epoch-3):epoch])))<change) 
          {
              flag_change_F=FALSE;
              reason=paste("Changes in F=beta*SCE + alpha * Ea + delta *Ed +alpha_c*c_a^2 + delta_c*c_d^2 in last 3 iterations less than",change,sep=" ");
          }
      }
      
      flag_C=TRUE
      flag_mu=mu<=mu_max
      while(flag_C & flag_mu)
      {
          Q=diag(c(rep(2*alpha+mu,npar1),rep(2*delta+mu,npar2),c(2*alpha_c+mu,2*delta_c+mu)))
	  tmp=as.vector(c(unlist(theta1),unlist(theta2),c(w1,w2))-solve(2*beta*H+Q,g))

	  theta_new1=list()
	  for(i in 1:neurons1)
	  {
	      theta_new1[[i]]=tmp[1:(2+p)]
	      tmp=tmp[-c(1:(2+p))]
	  }

          theta_new2=list()
          for(i in 1:neurons2)
          {
              theta_new2[[i]]=tmp[1:(2+p)]
	      tmp=tmp[-c(1:(2+p))]
          }
          w1_new=tmp[1]
          w2_new=tmp[2]	  
          
          #Paralell version
          p1_new=predictions.nn.C(vecX1,n,p,theta_new1,neurons1,cores)
          p2_new=predictions.nn.C(vecX2,n,q,theta_new2,neurons2,cores)
      
          e_new=y-w1_new*p1_new-w2_new*p2_new

	  Ed=sum(e_new^2)
	  E1=Ew(theta_new1)
          E2=Ew(theta_new2)
          E3=w1_new^2
          E4=w2_new^2

	  C_new=beta*Ed+alpha*E1+delta*E2+alpha_c*E3+delta_c*E4

	  if(verbose)
	  {
	    cat("C_new=",C_new,"\tEd=",Ed,"\tE1=",E1,"\tE2=",E2,"\tE3=",E3,"\tE4=",E4,"\n")
	  }
	  if(C_new<C)
	  {
	    mu=mu*mu_dec
            if (mu < 1e-20) mu = 1e-20;
	    flag_C=FALSE
	  }else{
	    mu=mu*mu_inc
	  }
	  if(verbose)
	  {
	    cat("mu=",mu,"\n")
	  } 
          flag_mu=mu<=mu_max
      }
      #Update all
     
      theta1=theta_new1
      theta2=theta_new2
      w1=w1_new
      w2=w2_new
      epoch=epoch+1
      Q1=diag(c(rep(2*alpha,npar1),rep(2*delta,npar2),c(2*alpha_c,2*delta_c)))
      d=diag(solve(2*beta*H+Q1))
      gamma1=npar1-2*alpha*sum(d[1:npar1])
      gamma2=npar2-2*delta*sum(d[(npar1+1):(npar1+npar2)])
      gamma3=1-2*alpha_c*d[npar1+npar2+1]; if(gamma3<0) gamma3=0.1
      gamma4=1-2*delta_c*d[npar1+npar2+2]; if(gamma4<0) gamma4=0.1

      alpha=gamma1/(2*E1)
      delta=gamma2/(2*E2)
      alpha_c=gamma3/(2*E3)
      delta_c=gamma4/(2*E4)
      
      beta=(n-gamma1-gamma2-gamma3-gamma4)/(2*Ed)
      
      if(Ed<0.01)
      {
         flag_change_Ed=FALSE
         reason="SCE <= 0.01";
      }
      if(verbose)
      {
	cat("gamma_a=",round(gamma1,4),"\tgamma_delta=",round(gamma2,4),"\tgamma_alpha_c=",round(gamma3,4),"\tgamma_delta_c=",round(gamma4,4),"\n")
	cat("alpha=",round(alpha,4),"\tdelta=",round(delta,4),"alpha_c=",round(alpha_c,4),"delta_c=",round(delta_c,4),"\tbeta=",round(beta,4),"\n")
      }
   }
   if((epoch-1)==epochs) reason="Maximum number of epochs reached";
   if(!flag_mu) reason="Maximum mu reached";
   if(!flag_gradient) reason="Minimum gradient reached"; 
   
   if(!verbose)
   {
      cat("gamma_a=",round(gamma1,4),"\tgamma_delta=",round(gamma2,4),"\tgamma_alpha_c=",round(gamma3,4),"\tgamma_delta_c=",round(gamma4,4),"\n")
      cat("alpha=",round(alpha,4),"\tdelta=",round(delta,4),"alpha_c=",round(alpha_c,4),"delta_c=",round(delta_c,4),"\tbeta=",round(beta,4),"\n")
   }
   
   out=list(theta1=theta1,theta2=theta2,c_a=w1,c_d=w2,alpha=alpha,beta=beta,
               delta=delta,alpha_c=alpha_c,delta_c=delta_c,
               E1=E1,E2=E2,reason=reason,epoch=epoch,F_history=F_history)
               
   class(out)="brnn.extended";

   return(out);
}

.First.lib <- function(lib, pkg) {
  library.dynam("brnn", pkg, lib)
}

##################################################################################################
.onAttach <- function(library, pkg)
{
  Rv <- R.Version()
  if(!exists("getRversion", baseenv()) || (getRversion() < "2.13.2"))
    stop("This package requires R 2.13.2 or later")
  assign(".brnn.home", file.path(library, pkg),
         pos=match("package:brnn", search()))
  brnn.version <- "0.1 (2012-07-31)"
  assign(".brnn.version", brnn.version, pos=match("package:brnn", search()))
  if(interactive())
  {
    packageStartupMessage(paste("Package 'brnn', ", brnn.version, ". ",sep=""),appendLF=TRUE)
    packageStartupMessage("Type 'help(brnn)' for summary information",appendLF=TRUE)
  }
  invisible()
}
##################################################################################################
