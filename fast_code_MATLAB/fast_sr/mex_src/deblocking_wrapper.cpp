#include "mex.h"
#include "image_op.h"
#include "deblocking.h"
using namespace std;

void deblocking_wrapper(int nlhs, mxArray* plhs[], int nrhs, const mxArray *prhs[]) {
	// result = bicubic_zzd(A)
	// printf("Entered deblocking wrapper\n");
	plhs[0] = mxDuplicateArray(prhs[0]);
	// printf("Output duplicated\n");

	int width = mxGetN(plhs[0]);
	int height = mxGetM(plhs[0]);
	vertical_deblock((int*)mxGetData(plhs[0]), width, height, height, 1);
}

// Entry function:
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	// Step 1: Check the functionality of bicubic interpolation
	deblocking_wrapper(nlhs, plhs, nrhs, prhs);
}