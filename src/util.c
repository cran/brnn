#include "util.h"

/*
 * tansig function, the evaluation of this function 
 * is faster than tanh and the derivative is the same!!! 
*/

double tansig(double x) 
{
    return(2.0/(1.0+exp(-2.0*x)) - 1.0); 
}

/*
 *sech function 
*/
double sech(double x)
{
    return(2.0*exp(x)/(exp(2.0*x)+1.0));
}
 
