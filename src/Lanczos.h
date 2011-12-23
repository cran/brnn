#include <stdlib.h>
#include <math.h>
#include <R.h>
#include <R_ext/Lapack.h>

void mgcv_trisymeig(double *d,double *g,double *v,int *n,int getvec,int descending);
void extreme_eigenvalues(double *A,double *U,double *D,int *n, int *m, int *lm,double *tol);
double Bai(double *A,int *n,double *lambdamin, double *lambdamax, double *tol, double *rz, int *col); 
