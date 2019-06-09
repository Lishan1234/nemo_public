#include "mex.h"
#include "image_op.h"
#include "deblocking.h"
using namespace std;

void bicubic_wrapper(int nlhs, mxArray* plhs[], int nrhs, const mxArray *prhs[]) {
	// result = bicubic_zzd(A, x, y, w, h)
	const mxArray* mxInput = prhs[0];
	float x = *((float*)mxGetData(prhs[1]));
	float y = *((float*)mxGetData(prhs[2]));
	int w = *((int*)mxGetData(prhs[3]));
	int h = *((int*)mxGetData(prhs[4]));

	plhs[0] = mxCreateDoubleMatrix(h, w, mxREAL);
	bicubic_interpolate<double>((double*)mxGetData(mxInput), (int)mxGetN(mxInput),
		(int)mxGetM(mxInput), x, y, w, h, (double*)mxGetData(plhs[0]));
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	// Step 1: Check the functionality of bicubic interpolation
	bicubic_wrapper(nlhs, plhs, nrhs, prhs);
}