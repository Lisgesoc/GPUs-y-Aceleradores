#ifndef ROUTINESGPU_H
#define ROUTINESGPU_H

#include <stdint.h>

void lane_assist_GPU(uint8_t *im, int height, int width,
	int *x1, int *y1, int *x2, int *y2, int *nlines, float *sin_table, float *cos_table);


void canny(uint8_t *img, uint8_t *image_out, int height, int width, float level);


#ifdef __CUDACC__
__global__ void NoiseReduct(uint8_t *im, float *NR, int height, int width);
__global__ void IntensityGrad(float *NR, float *Gx, float *Gy, float *G, float *phi, int height, int width);
__global__ void Edge(uint8_t *pedge, float *G, float *phi, int height, int width);
__global__ void Umbral(uint8_t *pedge, uint8_t *image_out, float *G, int height, int width, float level);
__global__ void houghTransform(uint8_t *im, int height, int width, uint32_t *accumulators, int accu_width, int accu_height, float *sin_table, float *cos_table, float hough_h);
__global__ void getLines(uint32_t *accumulators, int accu_width, int accu_height, int height, int width,int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines, float *sin_table, float *cos_table);
#endif

#endif

