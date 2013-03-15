/*  brnn/src/win/util_win.c by Paulino Perez Rodriguez
 *
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 or 3 of the License
 *  (at your option).
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  A copy of the GNU General Public License is available at
 *  http://www.r-project.org/Licenses/
 */

#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include "util.h"
#include "Lanczos.h"

SEXP predictions_nn(SEXP X, SEXP n, SEXP p, SEXP theta, SEXP neurons,SEXP yhat, SEXP reqCores)
{
   int i,j,k;
   double sum,z;
   int rows, columns, nneurons;
   double *pX, *ptheta, *pyhat;
   
   SEXP list;

   rows=INTEGER_VALUE(n);
   columns=INTEGER_VALUE(p);
   nneurons=INTEGER_VALUE(neurons);

   PROTECT(X=AS_NUMERIC(X));
   pX=NUMERIC_POINTER(X);

   PROTECT(theta=AS_NUMERIC(theta));
   ptheta=NUMERIC_POINTER(theta);


   PROTECT(yhat=AS_NUMERIC(yhat));
   pyhat=NUMERIC_POINTER(yhat);
   
   for(i=0;i<rows;i++)
   {
      		sum=0;
      		for(k=0;k<nneurons;k++)
      		{
	 		z=0;
	 		for(j=0;j<columns;j++)
	 		{
	    			z+=pX[i+(j*rows)]*ptheta[(columns+2)*k+j+2];
	 		}
	 		z+=ptheta[(columns+2)*k+1];      
	 		sum+=ptheta[(columns+2)*k]*tansig(z);
      		}
      		pyhat[i]=sum;
   }
   
   PROTECT(list=allocVector(VECSXP,1));
   SET_VECTOR_ELT(list,0,yhat);

   UNPROTECT(4);

   return(list);
}


//This function will calculate the Jocobian for the errors
SEXP jacobian(SEXP X, SEXP n, SEXP p, SEXP theta, SEXP neurons,SEXP J, SEXP reqCores)
{
   int i,j,k;
   double z,dtansig;
   double *pX;
   double *ptheta;
   double *pJ;
   int rows, columns, nneurons;

   SEXP list;

   rows=INTEGER_VALUE(n);
   columns=INTEGER_VALUE(p);
   nneurons=INTEGER_VALUE(neurons);
  
   PROTECT(X=AS_NUMERIC(X));
   pX=NUMERIC_POINTER(X);
   
   PROTECT(theta=AS_NUMERIC(theta));
   ptheta=NUMERIC_POINTER(theta);
   
   PROTECT(J=AS_NUMERIC(J));
   pJ=NUMERIC_POINTER(J);
   
  for(i=0; i<rows; i++)
  {
                //Rprintf("i=%d\n",i);
     		for(k=0; k<nneurons; k++)
     		{
	  		z=0;
	  		for(j=0;j<columns;j++)
	  		{
	      			z+=pX[i+(j*rows)]*ptheta[(columns+2)*k+j+2]; 
	  		}
	  		z+=ptheta[(columns+2)*k+1];
	  		dtansig=pow(sech(z),2.0);
	  
	  		/*
	  		 Derivative with respect to the weight
	  		*/
	  		pJ[i+(((columns+2)*k)*rows)]=-tansig(z);
	 
	  		/*
	  		Derivative with respect to the bias
	 		*/
	 
	 		pJ[i+(((columns+2)*k+1)*rows)]=-ptheta[(columns+2)*k]*dtansig;

	 		/*
	  		 Derivate with respect to the betas
	  		*/
	 		for(j=0; j<columns;j++)
	 		{
	     			pJ[i+(((columns+2)*k+j+2)*rows)]=-ptheta[(columns+2)*k]*dtansig*pX[i+(j*rows)];
	 		}
     		}
  }
  
  PROTECT(list=allocVector(VECSXP,1));
  SET_VECTOR_ELT(list,0,J);
  
  UNPROTECT(4);
   
  return(list);
}

SEXP estimate_trace(SEXP A, SEXP n, SEXP lambdamin, SEXP lambdamax, SEXP tol, SEXP samples, SEXP reqCores, SEXP rz, SEXP ans)
{ 
   int i;
   int nsamples;
   double lmin, lmax;
   double max_error;
   int rows;
   double *pA;
   double *pans;
   double *prz;
   double sum=0;
   
   
   SEXP list;
   
   nsamples=INTEGER_VALUE(samples);
   lmin=NUMERIC_VALUE(lambdamin);
   lmax=NUMERIC_VALUE(lambdamax);
   rows=INTEGER_VALUE(n);
   max_error=NUMERIC_VALUE(tol);
   
   PROTECT(A=AS_NUMERIC(A));
   pA=NUMERIC_POINTER(A);
   
   PROTECT(ans=AS_NUMERIC(ans));
   pans=NUMERIC_POINTER(ans);
   
   PROTECT(rz=AS_NUMERIC(rz));
   prz=NUMERIC_POINTER(rz);
   
   
   for(i=0; i<nsamples; i++)
   {
	sum+=Bai(pA,&rows,&lmin, &lmax, &max_error,prz,&i);
   }
   
   *pans=(sum/(nsamples));
   
   PROTECT(list=allocVector(VECSXP,1));
   SET_VECTOR_ELT(list,0,ans);
   
   UNPROTECT(4);
   
   return(list);
}
