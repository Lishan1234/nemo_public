#pragma once
#include <cmath>

#define CLAMP_BOUND(x, bound) ((x < 0)? 0:((x < bound)? x:(bound - 1)))
#define ACCESS_DATA(data, x, y, width, height) data[CLAMP_BOUND(x, width) * height + CLAMP_BOUND(y, height)]

float CubicHermite(float A, float B, float C, float D, float t)
{
	float a = -A / 2.0f + (3.0f*B) / 2.0f - (3.0f*C) / 2.0f + D / 2.0f;
	float b = A - (5.0f*B) / 2.0f + 2.0f*C - D / 2.0f;
	float c = -A / 2.0f + C / 2.0f;
	float d = B;

	return a*t*t*t + b*t*t + c*t + d;
}

// Bicubic Interpolation
template <typename T> void bicubic_interpolate(const T* input, int width, 
	int height, float x, float y, int w, int h, T* output) {
	// This function does NOT allocate the space for the output!

	float x_frac = x - floor(x);
	float y_frac = y - floor(y);

	for (int dx = 0; dx < w; dx++) {
		float x_now = x + dx;
		int x_int = int(floor(x_now));
		float y_now = y;
		int y_int = int(floor(y_now));

		float p00 = (float)ACCESS_DATA(input, x_int - 1, y_int - 1, width, height);
		float p10 = (float)ACCESS_DATA(input, x_int, y_int - 1, width, height);
		float p20 = (float)ACCESS_DATA(input, x_int + 1, y_int - 1, width, height);
		float p30 = (float)ACCESS_DATA(input, x_int + 2, y_int - 1, width, height);

		float v0 = CubicHermite(p00, p10, p20, p30, x_frac);

		float p01 = (float)ACCESS_DATA(input, x_int - 1, y_int, width, height);
		float p11 = (float)ACCESS_DATA(input, x_int, y_int, width, height);
		float p21 = (float)ACCESS_DATA(input, x_int + 1, y_int, width, height);
		float p31 = (float)ACCESS_DATA(input, x_int + 2, y_int, width, height);

		float v1 = CubicHermite(p01, p11, p21, p31, x_frac);

		float p02 = (float)ACCESS_DATA(input, x_int - 1, y_int + 1, width, height);
		float p12 = (float)ACCESS_DATA(input, x_int, y_int + 1, width, height);
		float p22 = (float)ACCESS_DATA(input, x_int + 1, y_int + 1, width, height);
		float p32 = (float)ACCESS_DATA(input, x_int + 2, y_int + 1, width, height);

		float v2 = CubicHermite(p02, p12, p22, p32, x_frac);

		for (int dy = 0; dy < h; dy++){
			float p03 = (float)ACCESS_DATA(input, x_int - 1, y_int + 2, width, height);
			float p13 = (float)ACCESS_DATA(input, x_int, y_int + 2, width, height);
			float p23 = (float)ACCESS_DATA(input, x_int + 1, y_int + 2, width, height);
			float p33 = (float)ACCESS_DATA(input, x_int + 2, y_int + 2, width, height);

			float v3 = CubicHermite(p03, p13, p23, p33, x_frac);

			// Actual interpolation
			float v_now = CubicHermite(v0, v1, v2, v3, y_frac);

			if (v_now < 0) v_now = 0;
			if (v_now > 255) v_now = 255;

			// Update
			y_int++;
			v0 = v1;
			v1 = v2;
			v2 = v3;
			ACCESS_DATA(output, dx, dy, w, h) = (T)v_now;
		}
	}
}

int mv_interpolate_line(int p1, int p2, int p3, int p4, int p5, int p6, int p7, int p8, int mv_frac_idx) {
	if (mv_frac_idx == 0) return p4 * 64;
	else if (mv_frac_idx == 1) {
		return -p1 + 4 * p2 - 10 * p3 + 58 * p4 + 17 * p5 - 5 * p6 + p7;
	}
	else if (mv_frac_idx == 2) {
		return -p1 + 4 * p2 - 11 * p3 + 40 * p4 + 40 * p5 - 11 * p6 + 4 * p7 - p8;
	}
	else if (mv_frac_idx == 3) {
		return p2 - 5 * p3 + 17 * p4 + 58 * p5 - 10 * p6 + 4 * p7 - p8;
	}
	else {
		printf("Error! Unrecognized mv_frac_idx: %d\n", mv_frac_idx);
		return 0;
	}
}

void mv_interpolate(int* ref_data, int x0, int y0, int mv_x_4m, int mv_y_4m, 
	int PU_width, int PU_height, int img_width, int img_height,
	int* out_data) {

	int xp0 = x0 + (mv_x_4m >> 2);
	int yp0 = y0 + (mv_y_4m >> 2);

	int mv_x_frac_idx = mv_x_4m - (mv_x_4m >> 2) * 4;
	int mv_y_frac_idx = mv_y_4m - (mv_y_4m >> 2) * 4;

	for (int dx = 0; dx < PU_width; dx++) {
		// Initialize
		int pre_compute[8];
		int x_idx = xp0 + dx;
		for (int i = 0; i < 7; i++) {
			int y_idx = yp0 + i;
			pre_compute[i] = mv_interpolate_line(
				ACCESS_DATA(ref_data, x_idx - 3, y_idx, img_width, img_height), 
				ACCESS_DATA(ref_data, x_idx - 2, y_idx, img_width, img_height),
				ACCESS_DATA(ref_data, x_idx - 1, y_idx, img_width, img_height),
				ACCESS_DATA(ref_data, x_idx, y_idx, img_width, img_height),
				ACCESS_DATA(ref_data, x_idx + 1, y_idx, img_width, img_height),
				ACCESS_DATA(ref_data, x_idx + 2, y_idx, img_width, img_height),
				ACCESS_DATA(ref_data, x_idx + 3, y_idx, img_width, img_height),
				ACCESS_DATA(ref_data, x_idx + 4, y_idx, img_width, img_height),
				mv_x_frac_idx);
		}
		for (int dy = 0; dy < PU_height; dy++) {
			int y_idx = yp0 + dy + 7;
			pre_compute[7] = mv_interpolate_line(
				ACCESS_DATA(ref_data, x_idx - 3, y_idx, img_width, img_height),
				ACCESS_DATA(ref_data, x_idx - 2, y_idx, img_width, img_height),
				ACCESS_DATA(ref_data, x_idx - 1, y_idx, img_width, img_height),
				ACCESS_DATA(ref_data, x_idx, y_idx, img_width, img_height),
				ACCESS_DATA(ref_data, x_idx + 1, y_idx, img_width, img_height),
				ACCESS_DATA(ref_data, x_idx + 2, y_idx, img_width, img_height),
				ACCESS_DATA(ref_data, x_idx + 3, y_idx, img_width, img_height),
				ACCESS_DATA(ref_data, x_idx + 4, y_idx, img_width, img_height),
				mv_x_frac_idx);

			int interp_val = mv_interpolate_line(pre_compute[0],
				pre_compute[1],
				pre_compute[2],
				pre_compute[3],
				pre_compute[4],
				pre_compute[5],
				pre_compute[6],
				pre_compute[7],
				mv_y_frac_idx);

			int final_val =((interp_val >> 6) + 32) >> 6;

			ACCESS_DATA(out_data, x0 + dx, y0 + dy, img_width, img_height) = final_val;

			for (int i = 0; i < 7; i++) {
				pre_compute[i] = pre_compute[i + 1];
			}
		}
	}


}