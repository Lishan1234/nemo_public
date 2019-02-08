#include "mex.h"
#include "image_op.h"
#include "deblocking.h"
using namespace std;

void bicubic_wrapper(int nlhs, mxArray* plhs[], int nrhs, const mxArray *prhs[]) {
	// result = bicubic_zzd(A, x, y, w, h)
	const mxArray* mxInput = prhs[0];
	float x = *((float*) mxGetData(prhs[1]));
	float y = *((float*) mxGetData(prhs[2]));
	int w = *((int*)mxGetData(prhs[3]));
	int h = *((int*)mxGetData(prhs[4]));

	plhs[0] = mxCreateDoubleMatrix(h, w, mxREAL);
	bicubic_interpolate<double>((double*) mxGetData(mxInput), (int) mxGetN(mxInput),
		(int) mxGetM(mxInput), x, y, w, h, (double*) mxGetData(plhs[0]));
}

void deblocking_wrapper(int nlhs, mxArray* plhs[], int nrhs, const mxArray *prhs[]) {
	// result = bicubic_zzd(A)
	// printf("Entered deblocking wrapper\n");
	plhs[0] = mxDuplicateArray(prhs[0]);
	// printf("Output duplicated\n");

	int width = mxGetN(plhs[0]);
	int height = mxGetM(plhs[0]);
	vertical_deblock((int*)mxGetData(plhs[0]), width, height, height, 1);
}

void deblocking_transpose_wrapper(int nlhs, mxArray* plhs[], int nrhs, const mxArray *prhs[]) {
	// result = bicubic_zzd(A)
	// printf("Entered deblocking tranpose wrapper\n");
	plhs[0] = mxDuplicateArray(prhs[0]);
	// printf("Output duplicated\n");

	int width = mxGetN(plhs[0]);
	int height = mxGetM(plhs[0]);
	vertical_deblock((int*)mxGetData(plhs[0]), height, width, 1, height);
}

void mv_compensate_wrapper(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
	// p_result = mv_compensate(img_data, x0, y0, PU_width, PU_height, mv_x, mv_y)

	// Parse parameters:
	int* ref_data = (int*) mxGetData(prhs[0]);
	int img_height = mxGetM(prhs[0]);
	int img_width = mxGetN(prhs[0]);

	int x0 = (int) mxGetScalar(prhs[1]);
	int y0 = (int)mxGetScalar(prhs[2]);

	int PU_width = (int)mxGetScalar(prhs[3]);
	int PU_height = (int)mxGetScalar(prhs[4]);

	int mv_x_4m = (int)mxGetScalar(prhs[5]);
	int mv_y_4m = (int)mxGetScalar(prhs[6]);

	plhs[0] = mxCreateNumericArray(2, mxGetDimensions(prhs[0]), mxINT32_CLASS, mxREAL);
	
	mv_interpolate(ref_data, x0, y0, mv_x_4m, mv_y_4m, PU_width, PU_height,
		img_width, img_height, (int*)mxGetData(plhs[0]));
}

// Entry function:
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	deblocking_wrapper(nlhs, plhs, nrhs, prhs);
}