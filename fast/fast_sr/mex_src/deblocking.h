#pragma once

#include <cmath>
#define BLK_WIDTH 8
#define HALF_BLK_WIDTH 4
#define QP 47
#define tc 10
#define beta 60

#define min(x, y) (((x) < (y))? (x):(y))
#define max(x, y) (((x) > (y))? (x):(y))

#define ACCESS_DATA_STRIDE(data, x, y, x_stride, y_stride) data[(x) * x_stride + (y) * y_stride]

void normal_filter_p0q0(int* data, int x, int y, int x_stride, int y_stride) {
	int p3 = ACCESS_DATA_STRIDE(data, x + 2, y, x_stride, y_stride);
	int p4 = ACCESS_DATA_STRIDE(data, x + 3, y, x_stride, y_stride);
	int p5 = ACCESS_DATA_STRIDE(data, x + 4, y, x_stride, y_stride);
	int p6 = ACCESS_DATA_STRIDE(data, x + 5, y, x_stride, y_stride);

	int d0 = (9 * (p5 - p4) - 3 * (p6 - p3) + 8) >> 4;
	d0 = min(max(-tc, d0), tc);

	// printf("p3 = %d, p4 = %d, p5 = %d, p6 = %d, d0 = %d\n", p3, p4, p5, p6, d0);
	ACCESS_DATA_STRIDE(data, x + 3, y, x_stride, y_stride) = min(max(p4 + d0, 0), 255);
	ACCESS_DATA_STRIDE(data, x + 4, y, x_stride, y_stride) = min(max(p5 - d0, 0), 255);
}

void normal_filter_p0q0p1q1(int* data, int x, int y, int x_stride, int y_stride) {
	int p2 = ACCESS_DATA_STRIDE(data, x + 1, y, x_stride, y_stride);
	int p3 = ACCESS_DATA_STRIDE(data, x + 2, y, x_stride, y_stride);
	int p4 = ACCESS_DATA_STRIDE(data, x + 3, y, x_stride, y_stride);
	int p5 = ACCESS_DATA_STRIDE(data, x + 4, y, x_stride, y_stride);
	int p6 = ACCESS_DATA_STRIDE(data, x + 5, y, x_stride, y_stride);
	int p7 = ACCESS_DATA_STRIDE(data, x + 6, y, x_stride, y_stride);

	// printf("p2 = %d, p3 = %d, p4 = %d, p5 = %d, p6 = %d, p7 = %d\n", p2, p3, p4, p5, p6, p7);

	int d0 = (9 * (p5 - p4) - 3 * (p6 - p3) + 8) >> 4;
	d0 = min(max(-tc, d0), tc);

	int dp1 = (((p2 + p4 + 1) >> 1) - p3 + d0 + 1) >> 1;
	int dq1 = (((p7 + p5 + 1) >> 1) - p6 - d0 + 1) >> 1;

	dp1 = min(max(-tc, dp1), tc / 2);
	dq1 = min(max(-tc, dq1), tc / 2);

	// printf("d0 = %d, dp1 = %d, dq1 = %d\n", d0, dp1, dq1);
	ACCESS_DATA_STRIDE(data, x + 2, y, x_stride, y_stride) = min(max(p3 + dp1, 0), 255);
	ACCESS_DATA_STRIDE(data, x + 3, y, x_stride, y_stride) = min(max(p4 + d0, 0), 255);

	ACCESS_DATA_STRIDE(data, x + 4, y, x_stride, y_stride) = min(max(p5 - d0, 0), 255);
	ACCESS_DATA_STRIDE(data, x + 5, y, x_stride, y_stride) = min(max(p6 + dq1, 0), 255);
}

void strong_filter(int* data, int x, int y, int x_stride, int y_stride) {
	int p1 = ACCESS_DATA_STRIDE(data, x, y, x_stride, y_stride);
	int p2 = ACCESS_DATA_STRIDE(data, x + 1, y, x_stride, y_stride);
	int p3 = ACCESS_DATA_STRIDE(data, x + 2, y, x_stride, y_stride);
	int p4 = ACCESS_DATA_STRIDE(data, x + 3, y, x_stride, y_stride);
	int p5 = ACCESS_DATA_STRIDE(data, x + 4, y, x_stride, y_stride);
	int p6 = ACCESS_DATA_STRIDE(data, x + 5, y, x_stride, y_stride);
	int p7 = ACCESS_DATA_STRIDE(data, x + 6, y, x_stride, y_stride);
	int p8 = ACCESS_DATA_STRIDE(data, x + 7, y, x_stride, y_stride);

	// printf("p1 = %d, p2 = %d, p3 = %d, p4 = %d, p5 = %d, p6 = %d, p7 = %d, p8 = %d\n", 
	//	p1, p2, p3, p4, p5, p6, p7, p8);

	int c = 2 * tc;
	
	int dp0 = (p2 + 2 * p3 - 6 * p4 + 2 * p5 + p6 + 4) >> 3;
	int dp1 = (p2 - 3 * p3 + p4 + p5 + 2) >> 2;
	int dp2 = (2 * p1 - 5 * p2 + p3 + p4 + p5 + 4) >> 3;

	int dq0 = (p7 + 2 * p6 - 6 * p5 + 2 * p4 + p3 + 4) >> 3;
	int dq1 = (p7 - 3 * p6 + p5 + p4 + 2) >> 2;
	int dq2 = (2 * p8 - 5 * p7 + p6 + p5 + p4 + 4) >> 3;

	dp0 = min(max(-c, dp0), c);
	dp1 = min(max(-c, dp1), c);
	dp2 = min(max(-c, dp2), c);

	dq0 = min(max(-c, dq0), c);
	dq1 = min(max(-c, dq1), c);
	dq2 = min(max(-c, dq2), c);

	// printf("dp0 = %d, dp1 = %d, dp2 = %d, dq0 = %d, dq1 = %d, dq2 = %d\n",
	//	dp0, dp1, dp2, dq0, dq1, dq2);

	// ACCESS_DATA_STRIDE(data, x, y, x_stride, y_stride) = min(max(p1, 0), 255);

	ACCESS_DATA_STRIDE(data, x + 1, y, x_stride, y_stride) = min(max(p2 + dp2, 0), 255);
	ACCESS_DATA_STRIDE(data, x + 2, y, x_stride, y_stride) = min(max(p3 + dp1, 0), 255);
	ACCESS_DATA_STRIDE(data, x + 3, y, x_stride, y_stride) = min(max(p4 + dp0, 0), 255);

	ACCESS_DATA_STRIDE(data, x + 4, y, x_stride, y_stride) = min(max(p5 + dq0, 0), 255);
	ACCESS_DATA_STRIDE(data, x + 5, y, x_stride, y_stride) = min(max(p6 + dq1, 0), 255);
	ACCESS_DATA_STRIDE(data, x + 6, y, x_stride, y_stride) = min(max(p7 + dq2, 0), 255);

	// ACCESS_DATA_STRIDE(data, x + 7, y, x_stride, y_stride) = min(max(p8, 0), 255);
}

/*void get_beta_tc(int nQP, int& beta, int& tc) {
	switch (nQP) {
	case 37:
		beta = 40;
		tc = 5;
		break;
	case 42:
		beta = 45;
		tc = 8;
		break;
	case 47:
		beta = 60;
		tc = 10;
		break;
	default:
		beta = 60;
		tc = 10;
		break;
	}
}*/

void vertical_deblock(int* data, const int width, const int height,
	const int x_stride, const int y_stride) {
	// printf("Entering deblocking function!\n");
	// printf("x_stride = %d, y_stride = %d\n", x_stride, y_stride);
	int N_r = height / HALF_BLK_WIDTH;
	int N_c = width / BLK_WIDTH - 1;

	// Output the patch
/*	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			printf("%d ", ACCESS_DATA_STRIDE(data, x, y, x_stride, y_stride));
		}
		printf("\n");
	} */

	for (int c_idx = 0; c_idx < N_c; c_idx++) {
		for (int r_idx = 0; r_idx < N_r; r_idx++) {
			// printf("c = %d, r = %d\n", c_idx, r_idx);
			int y0 = r_idx * HALF_BLK_WIDTH;
			int x0 = HALF_BLK_WIDTH + c_idx * BLK_WIDTH;

			// Judge deblocking or not

			// printf("x0 = %d, y0 = %d, data(5, 0) = %d\n", x0 + 1, y0, ACCESS_DATA_STRIDE(data, x0 + 1, 0, x_stride, y_stride));
			int p12 = ACCESS_DATA_STRIDE(data, x0 + 1, y0, x_stride, y_stride);
			int p13 = ACCESS_DATA_STRIDE(data, x0 + 2, y0, x_stride, y_stride);
			int p14 = ACCESS_DATA_STRIDE(data, x0 + 3, y0, x_stride, y_stride);
			// printf("p12 = %d, p13 = %d, p14 = %d\n", p12, p13, p14);

			int p_l1 = abs(p12 - 2 * p13 + p14);

			int p15 = ACCESS_DATA_STRIDE(data, x0 + 4, y0, x_stride, y_stride);
			int p16 = ACCESS_DATA_STRIDE(data, x0 + 5, y0, x_stride, y_stride);
			int p17 = ACCESS_DATA_STRIDE(data, x0 + 6, y0, x_stride, y_stride);
			// printf("p15 = %d, p16 = %d, p17 = %d\n", p15, p16, p17);

			int q_l1 = abs(p15 - 2 * p16 + p17);

			int p42 = ACCESS_DATA_STRIDE(data, x0 + 1, y0 + 3, x_stride, y_stride);
			int p43 = ACCESS_DATA_STRIDE(data, x0 + 2, y0 + 3, x_stride, y_stride);
			int p44 = ACCESS_DATA_STRIDE(data, x0 + 3, y0 + 3, x_stride, y_stride);
			// printf("p42 = %d, p43 = %d, p44 = %d\n", p42, p43, p44);

			int p_l4 = abs(p42 - 2 * p43 + p44);

			int p45 = ACCESS_DATA_STRIDE(data, x0 + 4, y0 + 3, x_stride, y_stride);
			int p46 = ACCESS_DATA_STRIDE(data, x0 + 5, y0 + 3, x_stride, y_stride);
			int p47 = ACCESS_DATA_STRIDE(data, x0 + 6, y0 + 3, x_stride, y_stride);
			// printf("p45 = %d, p46 = %d, p47 = %d\n", p45, p46, p47);

			int q_l4 = abs(p45 - 2 * p46 + p47);

			// printf("x = %d, y = %d, p_l1 = %d, q_l1 = %d, p_l4 = %d, q_l4 = %d\n",
			//	x0, y0, p_l1, q_l1, p_l4, q_l4);

			if ((p_l1 + q_l1 + p_l4 + q_l4) > beta)
				// NO DEBLOCKING AT ALL!
				continue;
			else {
				// Judge Strong or normal filter
				bool b_strong_filter = true;
				// b1
				if ((p_l1 + q_l1 <= beta / 8) && (p_l4 + q_l4) <= (beta / 8)) {
					// b2
					int p11 = ACCESS_DATA_STRIDE(data, x0, y0, x_stride, y_stride);
					int p18 = ACCESS_DATA_STRIDE(data, x0 + 7, y0, x_stride, y_stride);
					if (abs(p11 - p14) + abs(p15 - p18) <= beta / 8) {
						// b3
						int p41 = ACCESS_DATA_STRIDE(data, x0, y0 + 3, x_stride, y_stride);
						int p48 = ACCESS_DATA_STRIDE(data, x0 + 7, y0 + 3, x_stride, y_stride);
						if (abs(p41 - p44) + abs(p45 - p48) <= beta / 8) {
							// b4
							if ((abs(p14 - p15) < 5 * tc / 2) && (abs(p44 - p45) < 5 * tc / 2)) {
								b_strong_filter = true;
							}
							else {
								b_strong_filter = false;
							}
						}
						else {
							b_strong_filter = false;
						}
					}
					else {
						b_strong_filter = false;
					}

				}
				else {
					b_strong_filter = false;
				}

				if (b_strong_filter) {
					// Apply strong filter
					// printf("Strong filter applied!\n");
					for (int dy = 0; dy < 4; dy++) {
						strong_filter(data, x0, y0 + dy, x_stride, y_stride);
					}
				}
				else {
					// Apply weak filter
					if ((p_l1 + q_l1 <= beta * 3 / 8) && (p_l4 + q_l4 <= beta * 3 / 8)) {
						// printf("Weak filter 1 applied!\n");
						for (int dy = 0; dy < 4; dy++) {
							normal_filter_p0q0p1q1(data, x0, y0 + dy, x_stride, y_stride);
						}
					}
					else {
						// printf("Weak filter 2 applied!\n");
						for (int dy = 0; dy < 4; dy++) {
							normal_filter_p0q0(data, x0, y0 + dy, x_stride, y_stride);
						}
					}

				}
			}

		}
	}
}