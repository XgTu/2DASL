#include <mex.h>
#include <matrix.h>
#include "Tnorm_Vnorm.h"

void mexFunction(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[])
{
	int x,y,n;

	double* normt;
	double* tri;
	int ntri;
	int nver;

	double* normv;
	
	normt = mxGetPr(prhs[0]);
	tri = mxGetPr(prhs[1]);
	ntri = (int)*mxGetPr(prhs[2]);
	nver = (int)*mxGetPr(prhs[3]);
	
	plhs[0] = mxCreateDoubleMatrix(3, nver, mxREAL);
	normv = mxGetPr(plhs[0]);
    
	TNorm2VNorm(normv, nver, normt, tri, ntri);

}