#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include "util.h"
#include "Lanczos.h"

#ifdef SUPPORT_OPENMP
  #include <omp.h>
  #define CSTACK_DEFNS 7
  #include "Rinterface.h"
#endif

SEXP predictions_nn(SEXP X, SEXP n, SEXP p, SEXP theta, SEXP neurons,SEXP yhat, SEXP reqCores)
{
   int i,j,k;
   double sum,z;
   int rows, columns, nneurons;
   int useCores, haveCores;
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

   /*
   Set the number of threads
   */
   
   #ifdef SUPPORT_OPENMP     
     R_CStackLimit=(uintptr_t)-1;
     useCores=INTEGER_VALUE(reqCores);
     haveCores=omp_get_num_procs();
     if(useCores<=0 || useCores>haveCores) useCores=haveCores;
     omp_set_num_threads(useCores);   
   #endif

   #pragma omp parallel private(j,k,z,sum) 
   {
        #pragma omp for schedule(static)
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
   int useCores, haveCores;
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
   
   /*
   Set the number of threads
   */

   #ifdef SUPPORT_OPENMP
     R_CStackLimit=(uintptr_t)-1;
     useCores=INTEGER_VALUE(reqCores);
     haveCores=omp_get_num_procs();
     if(useCores<=0 || useCores>haveCores) useCores=haveCores;
     omp_set_num_threads(useCores); 
   #endif

   #pragma omp parallel private(j,k,z,dtansig) 
   {
        #pragma omp for schedule(static)
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
   }
  
   PROTECT(list=allocVector(VECSXP,1));
   SET_VECTOR_ELT(list,0,J);
   
   UNPROTECT(4);
   
   return(list);
}

SEXP estimate_trace(SEXP A, SEXP n, SEXP lambdamin, SEXP lambdamax, SEXP tol, SEXP samples, SEXP reqCores, SEXP rz, SEXP ans)
{ 
   int useCores, haveCores;
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
   useCores=INTEGER_VALUE(reqCores);
   
   PROTECT(A=AS_NUMERIC(A));
   pA=NUMERIC_POINTER(A);
   
   PROTECT(ans=AS_NUMERIC(ans));
   pans=NUMERIC_POINTER(ans);
   
   PROTECT(rz=AS_NUMERIC(rz));
   prz=NUMERIC_POINTER(rz);
   
   /*
   Set the number of threads
   */
   
   #ifdef SUPPORT_OPENMP
     R_CStackLimit=(uintptr_t)-1;
     haveCores=omp_get_num_procs();
     if(useCores<=0 || useCores>haveCores) useCores=haveCores;
     omp_set_num_threads(useCores);
   #endif 
   
   #pragma omp parallel
   {
      /*Starts the work sharing construct*/
      #pragma omp for reduction(+:sum) schedule(static)
      for(i=0; i<nsamples; i++)
      {
	sum+=Bai(pA,&rows,&lmin, &lmax, &max_error,prz,&i);
      }
   }
   
   *pans=(sum/(nsamples));
   
   PROTECT(list=allocVector(VECSXP,1));
   SET_VECTOR_ELT(list,0,ans);
   
   UNPROTECT(4);
   
   return(list);
}
