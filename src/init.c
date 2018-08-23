/*
This file was autogenerated with the output of the following R Code:

tools::package_native_routine_registration_skeleton("/home/pperez/brnn")

*/

#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME: 
 *    Check these declarations against the C/Fortran source code.
 *    */

/* .C calls */
extern void extreme_eigenvalues(void *, void *, void *, void *, void *, void *, void *);

/* .Call calls */
extern SEXP estimate_trace(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP jacobian_(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP predictions_nn(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CMethodDef CEntries[] = {
    {"extreme_eigenvalues", (DL_FUNC) &extreme_eigenvalues, 7},
    {NULL, NULL, 0}
};

static const R_CallMethodDef CallEntries[] = {
    {"estimate_trace", (DL_FUNC) &estimate_trace, 9},
    {"jacobian_",      (DL_FUNC) &jacobian_,      7},
    {"predictions_nn", (DL_FUNC) &predictions_nn, 7},
    {NULL, NULL, 0}
};

void R_init_brnn(DllInfo *dll)
{
    R_registerRoutines(dll, CEntries, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

